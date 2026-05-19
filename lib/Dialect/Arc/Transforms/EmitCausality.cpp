//===- EmitCausality.cpp - Inject path-sensitive causality tracing --------===//
//
// Runs at the END of populateArcStateLoweringPipeline, after LowerState +
// InlineArcs + MergeIfs.  At that point:
//   - arc.alloc_state / arc.root_input / arc.root_output ops hold state handles
//     with "name" attributes preserved from the original FIRRTL signal names.
//   - arc.state_write $handle, $val    -- write a state (our injection site)
//   - arc.state_read  $handle          -- read a state (predecessor source)
//   - comb.mux $cond, $true, $false    -- mux is still present (MuxToControlFlow
//     is commented out in the pipeline); used for path-sensitive branching.
//
// The pass does two things:
//   1. (Compile-time) Enumerate all state handles, assign sequential IDs,
//      write __signal_index.json to causalityDir/.
//   2. (Instrumentation) For every arc.state_write, emit one causality_begin /
//      causality_commit pair around a recursive walk of the written value's
//      combinational expression.  The walk emits causality_add_pred calls at
//      arc.state_read leaves (DATA role) and inserts scf.if guards at every
//      comb.mux so that only the runtime-active branch's state reads become
//      predecessors of the event.  Mux conditions that are direct state reads
//      also get a CONTROL_GUARD predecessor.
//
// Every state write is instrumented unconditionally so that backward slicing
// can follow predecessor chains transitively through the full circuit.  The
// --causality-sinks command-line option is currently ignored (it used to
// restrict instrumentation to a hand-picked subset of observable signals).
//
// Registration: this pass is registered manually in arcilator.cpp (not via
// ArcPasses.td) so that touching ArcPasses.td and triggering a full rebuild
// of all Arc transforms is avoided.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/EmitCausalityPass.h"

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace arc;
using namespace mlir;

// ---------------------------------------------------------------------------
// Debug logging: set CAUSALITY_DEBUG=1 in the environment before running
// arcilator to get a per-write-site trace of every operation visited, every
// guard emitted, and every add_pred call generated.
// ---------------------------------------------------------------------------
static bool causalityDebug() {
  static int cached = -1;
  if (cached == -1)
    cached = (::getenv("CAUSALITY_DEBUG") != nullptr) ? 1 : 0;
  return cached == 1;
}

// Indent level for nested walkValue calls.
static thread_local int gWalkDepth = 0;

static void dbg(const llvm::Twine &msg) {
  if (!causalityDebug()) return;
  for (int i = 0; i < gWalkDepth; ++i) llvm::errs() << "  ";
  llvm::errs() << "[causality] " << msg << "\n";
}

namespace {

// Predecessor roles (must match causality_runtime.cc)
static constexpr int8_t ROLE_DATA = 0;
static constexpr int8_t ROLE_CONTROL_GUARD = 1;
static constexpr int8_t ROLE_PRIOR_STATE = 3;

// -----------------------------------------------------------------------
// Utility: get the signal name from a state handle Value.
// -----------------------------------------------------------------------
static StringRef getStateName(Value state) {
  Operation *defOp = state.getDefiningOp();
  if (!defOp)
    return {};
  if (auto allocOp = dyn_cast<AllocStateOp>(defOp)) {
    if (auto nameAttr = allocOp->getAttrOfType<StringAttr>("name"))
      return nameAttr.getValue();
    return {};
  }
  if (auto rootIn = dyn_cast<RootInputOp>(defOp))
    return rootIn.getName();
  if (auto rootOut = dyn_cast<RootOutputOp>(defOp))
    return rootOut.getName();
  return {};
}

// -----------------------------------------------------------------------
// Resolve a Value to a direct arc.state_read storage handle (or null).
// -----------------------------------------------------------------------
static Value resolveDirectState(Value val) {
  if (auto readOp = val.getDefiningOp<StateReadOp>())
    return readOp.getState();
  return {};
}

// -----------------------------------------------------------------------
// The pass — defined with mlir::PassWrapper to avoid modifying ArcPasses.td
// (which would invalidate and recompile all 30+ Arc transform files).
// -----------------------------------------------------------------------
class EmitCausalityPass
    : public PassWrapper<EmitCausalityPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitCausalityPass)

  EmitCausalityPass() = default;
  EmitCausalityPass(const EmitCausalityPass &) = default;
  explicit EmitCausalityPass(circt::arc::EmitCausalityPassOptions opts)
      : opts(std::move(opts)) {}

  StringRef getArgument() const final { return "arc-emit-causality"; }
  StringRef getDescription() const final {
    return "Emit __signal_index.json and inject path-sensitive causality "
           "calls at arc.state_write sites";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, scf::SCFDialect, arith::ArithDialect>();
  }

  void runOnOperation() override;

private:
  circt::arc::EmitCausalityPassOptions opts;

  ModuleOp module;
  MLIRContext *ctx = nullptr;

  // State handle Value → 1-indexed signal ID
  DenseMap<Value, int64_t> storageToId;

  // arc.state_write Operation* → 1-indexed write-site ID
  DenseMap<Operation *, int64_t> writeOpToSiteId;

  // External function declarations added to the module
  func::FuncOp beginDecl, addPredDecl, commitDecl;

  void enumSignals();
  void enumWriteSites();
  void writeSignalIndex();
  void declareRuntimeFuncs();
  void injectForWrite(OpBuilder &b, StateWriteOp writeOp, int64_t sinkId);
  void walkValue(OpBuilder &b, Value val, DenseSet<Value> &visited);

  void emitBegin(OpBuilder &b, Location loc, int64_t sinkId, int64_t wsid,
                 Value newVal, int32_t numBits);
  void emitAddPred(OpBuilder &b, Location loc, int64_t predId, int8_t role,
                   int8_t delta);
  void emitCommit(OpBuilder &b, Location loc);

  Value toI64(OpBuilder &b, Location loc, Value val);
  Value cI64(OpBuilder &b, Location loc, int64_t v);
  Value cI32(OpBuilder &b, Location loc, int32_t v);
  Value cI8(OpBuilder &b, Location loc, int8_t v);

  int64_t lookupId(Value stateHandle) {
    auto it = storageToId.find(stateHandle);
    return (it == storageToId.end()) ? -1 : it->second;
  }

  func::FuncOp getOrDeclare(StringRef name, FunctionType ft);
};

} // namespace

// =======================================================================
// Factory
// =======================================================================

std::unique_ptr<Pass>
circt::arc::createEmitCausalityPass(EmitCausalityPassOptions opts) {
  return std::make_unique<EmitCausalityPass>(std::move(opts));
}

// =======================================================================
// runOnOperation
// =======================================================================

void EmitCausalityPass::runOnOperation() {
  module = getOperation();
  ctx = &getContext();

  if (opts.causalityDir.empty())
    return;

  enumSignals();
  enumWriteSites();
  writeSignalIndex();
  declareRuntimeFuncs();

  // Instrument ALL write sites, not just named sinks.
  // Every write site must emit a causality event so that backward slicing
  // can follow predecessors transitively through the full circuit.
  SmallVector<StateWriteOp> writes;
  module.walk([&](StateWriteOp op) { writes.push_back(op); });

  for (auto writeOp : writes) {
    int64_t sinkId = lookupId(writeOp.getState());
    if (sinkId == -1)
      continue;
    OpBuilder builder(writeOp);
    injectForWrite(builder, writeOp, sinkId);
  }
}

// =======================================================================
// Phase 1: enumerate state-holding ops and assign sequential IDs
// =======================================================================

void EmitCausalityPass::enumSignals() {
  int64_t counter = 1;
  module.walk([&](Operation *op) {
    Value stateResult;
    if (auto v = dyn_cast<RootInputOp>(op))
      stateResult = v.getState();
    else if (auto v = dyn_cast<RootOutputOp>(op))
      stateResult = v.getState();
    else if (auto v = dyn_cast<AllocStateOp>(op))
      stateResult = v.getState();
    if (stateResult)
      storageToId[stateResult] = counter++;
  });
}

// =======================================================================
// Phase 1b: enumerate arc.state_write ops and assign write-site IDs
// =======================================================================

void EmitCausalityPass::enumWriteSites() {
  int64_t counter = 1;
  module.walk([&](StateWriteOp op) {
    writeOpToSiteId[op.getOperation()] = counter++;
  });
}

// =======================================================================
// Phase 2: write __signal_index.json
// =======================================================================

void EmitCausalityPass::writeSignalIndex() {
  llvm::json::Array signals;
  int64_t counter = 1;

  module.walk([&](Operation *op) {
    Value stateResult;
    StringRef name;
    if (auto v = dyn_cast<RootInputOp>(op)) {
      stateResult = v.getState();
      name = v.getName();
    } else if (auto v = dyn_cast<RootOutputOp>(op)) {
      stateResult = v.getState();
      name = v.getName();
    } else if (auto v = dyn_cast<AllocStateOp>(op)) {
      stateResult = v.getState();
      if (auto a = v->getAttrOfType<StringAttr>("name"))
        name = a.getValue();
    }
    if (!stateResult)
      return;

    llvm::json::Object entry;
    entry["id"] = counter++;
    entry["name"] = name.empty() ? "reg_" + std::to_string(counter - 1)
                                 : name.str();
    signals.push_back(llvm::json::Value(std::move(entry)));
  });

  llvm::json::Object root;
  root["format"] = "trace_causality_signal_index";
  root["signals"] = llvm::json::Value(std::move(signals));

  // Write-site table: one entry per arc.state_write op, using the same
  // deterministic walk order as enumWriteSites().  write_site_id is unique
  // per write op; signal_id/signal_name identify the target register.
  llvm::json::Array writeSites;
  module.walk([&](StateWriteOp op) {
    int64_t wsid = writeOpToSiteId.lookup(op.getOperation());
    int64_t sigId = lookupId(op.getState());
    StringRef name = getStateName(op.getState());
    llvm::json::Object e;
    e["write_site_id"] = wsid;
    if (sigId != -1)
      e["signal_id"] = sigId;
    if (!name.empty())
      e["signal_name"] = name.str();
    writeSites.push_back(llvm::json::Value(std::move(e)));
  });
  root["write_sites"] = llvm::json::Value(std::move(writeSites));

  std::string outPath = opts.causalityDir + "/__signal_index.json";
  std::error_code ec;
  llvm::raw_fd_ostream os(outPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    module.emitError("EmitCausality: cannot open ")
        << outPath << ": " << ec.message();
    signalPassFailure();
    return;
  }
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(root)));
}

// =======================================================================
// Phase 3: declare external runtime functions
// =======================================================================

func::FuncOp EmitCausalityPass::getOrDeclare(StringRef name, FunctionType ft) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder b(ctx);
  b.setInsertionPointToEnd(module.getBody());
  auto funcOp = b.create<func::FuncOp>(module.getLoc(), name, ft);
  // External declarations (empty body) must have private visibility.
  funcOp.setVisibility(mlir::SymbolTable::Visibility::Private);
  return funcOp;
}

void EmitCausalityPass::declareRuntimeFuncs() {
  auto i64 = IntegerType::get(ctx, 64);
  auto i32 = IntegerType::get(ctx, 32);
  auto i8 = IntegerType::get(ctx, 8);

  beginDecl = getOrDeclare("causality_begin",
                            FunctionType::get(ctx, {i64, i64, i64, i32}, {}));
  addPredDecl = getOrDeclare("causality_add_pred",
                              FunctionType::get(ctx, {i64, i8, i8}, {}));
  commitDecl =
      getOrDeclare("causality_commit", FunctionType::get(ctx, {}, {}));
}

// =======================================================================
// Phase 4: inject causality instrumentation
// =======================================================================

void EmitCausalityPass::injectForWrite(OpBuilder &builder, StateWriteOp writeOp,
                                        int64_t sinkId) {
  Location loc = writeOp.getLoc();
  Value writtenVal = writeOp.getValue();
  int32_t numBits = 0;
  if (auto itype = dyn_cast<IntegerType>(writtenVal.getType()))
    numBits = static_cast<int32_t>(itype.getWidth());

  int64_t wsid = writeOpToSiteId.lookup(writeOp.getOperation());
  StringRef sinkName = getStateName(writeOp.getState());

  dbg("=== WriteOp sink_id=" + llvm::Twine(sinkId) +
      " (" + (sinkName.empty() ? "?" : sinkName) + ")" +
      " ws_id=" + llvm::Twine(wsid) +
      " bits=" + llvm::Twine(numBits));

  emitBegin(builder, loc, sinkId, wsid, writtenVal, numBits);
  DenseSet<Value> visited;
  gWalkDepth = 1;
  walkValue(builder, writtenVal, visited);
  gWalkDepth = 0;
  emitCommit(builder, loc);
}

// -----------------------------------------------------------------------
// walkValue: recursively descend through a combinational expression, emitting
//   - DATA add_pred at every arc.state_read leaf,
//   - scf.if-guarded sub-walks at every comb.mux (only the executed branch's
//     state reads become predecessors of the current event),
//   - CONTROL_GUARD add_pred when a mux condition is a direct state read.
//
// The `visited` set prevents redundant descent into CSE-shared subexpressions
// within a single straight-line walk. It is passed by reference so insertions
// accumulate across sibling operands of the same parent op. At each scf.if
// boundary we COPY the current visited set into the then/else branches: this
// (a) inherits dedup state from the parent context (a state read already
// emitted unconditionally before the scf.if must not be re-emitted inside the
// branch, since the parent's add_pred always fires), and (b) keeps the two
// branches independent of each other (a state read referenced in both
// branches must be emitted in each, because only one branch's IR runs at
// runtime).
//
// OVERAPPROXIMATION (known, minor):
//   - comb.extract: a fault in bits outside the extracted range cannot
//     propagate. Fixing requires per-bit range tracking across the walk —
//     non-trivial and rare in practice.
//   - comb.mux conditions that are complex sub-expressions (not direct
//     arc.state_reads) are walked as DATA, missing CONTROL_GUARD tagging for
//     the state reads inside the condition. Cosmetic: affects slicer edge
//     coloring only, not injection site selection.
// -----------------------------------------------------------------------
void EmitCausalityPass::walkValue(OpBuilder &builder, Value val,
                                   DenseSet<Value> &visited) {
  if (!visited.insert(val).second) {
    dbg("(already visited, skip)");
    return;
  }

  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    dbg("block-arg / constant (no defOp), skip");
    return;
  }

  // Stop at state-write boundaries: a written register is a separate event,
  // not a predecessor reached through combinational logic.
  if (isa<StateWriteOp>(defOp)) {
    dbg("stop at StateWriteOp boundary");
    return;
  }

  // Leaf: arc.state_read becomes a DATA predecessor.
  if (auto readOp = dyn_cast<StateReadOp>(defOp)) {
    int64_t predId = lookupId(readOp.getState());
    StringRef name = getStateName(readOp.getState());
    dbg("state_read id=" + llvm::Twine(predId) +
        " (" + (name.empty() ? "?" : name) + ")" +
        " -> add_pred(DATA)");
    if (predId != -1)
      emitAddPred(builder, readOp.getLoc(), predId, ROLE_DATA, -1);
    return;
  }

  // Path-sensitive split: comb.mux executes only one of its data operands.
  if (auto muxOp = dyn_cast<comb::MuxOp>(defOp)) {
    Value cond = muxOp.getCond();
    Location muxLoc = muxOp.getLoc();

    dbg("comb.mux  -> walk cond unconditionally");
    ++gWalkDepth;
    walkValue(builder, cond, visited);
    --gWalkDepth;

    if (auto s = resolveDirectState(cond)) {
      int64_t cid = lookupId(s);
      StringRef name = getStateName(s);
      dbg("comb.mux  cond is direct state_read id=" + llvm::Twine(cid) +
          " (" + (name.empty() ? "?" : name) + ") -> add_pred(CONTROL_GUARD)");
      if (cid != -1)
        emitAddPred(builder, muxLoc, cid, ROLE_CONTROL_GUARD, -1);
    }

    dbg("comb.mux  emit scf.if; then-branch walks trueVal, else-branch walks falseVal");
    auto ifOp =
        builder.create<scf::IfOp>(muxLoc, cond, /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(ifOp.thenBlock()->getTerminator());
      DenseSet<Value> thenVisited = visited;
      ++gWalkDepth;
      dbg("comb.mux  [then]:");
      walkValue(builder, muxOp.getTrueValue(), thenVisited);
      --gWalkDepth;
    }
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(ifOp.elseBlock()->getTerminator());
      DenseSet<Value> elseVisited = visited;
      ++gWalkDepth;
      dbg("comb.mux  [else]:");
      walkValue(builder, muxOp.getFalseValue(), elseVisited);
      --gWalkDepth;
    }
    return;
  }

  // Path-sensitive: comb.and / comb.or short-circuit masking.
  //
  // A single-bit fault in operand i propagates through a gate only when the
  // other operands do not already force the output to a fixed value:
  //   AND: operand i matters only when AND(all others) != 0
  //         (if any other operand is 0, the output is stuck at 0 regardless)
  //   OR:  operand i matters only when OR(all others) == 0
  //         (if any other operand is 1, the output is stuck at 1 regardless)
  //
  // For each operand we compute the AND/OR of the remaining ones at compile
  // time (as an arith expression over the already-live SSA values), then emit
  // a runtime scf.if guard before recursing into that operand's sub-tree.
  //
  // Corner cases:
  //   - N=1: single operand always propagates; walk unconditionally.
  //   - AND, two operands both 0 (or OR, two operands both 1): guards for
  //     both operands evaluate to false at runtime, so zero add_pred calls
  //     are emitted. This is correct: no single-bit flip can change the
  //     output in that configuration.
  if (isa<comb::AndOp, comb::OrOp>(defOp)) {
    const bool isAnd = isa<comb::AndOp>(defOp);
    OperandRange operands = defOp->getOperands();
    Location loc = defOp->getLoc();

    dbg(llvm::Twine(isAnd ? "comb.and" : "comb.or") +
        "  " + llvm::Twine(operands.size()) + " operands, emitting guards");

    for (unsigned i = 0; i < operands.size(); ++i) {
      Value guardExpr;
      for (unsigned j = 0; j < operands.size(); ++j) {
        if (j == i)
          continue;
        if (!guardExpr) {
          guardExpr = operands[j];
        } else if (isAnd) {
          guardExpr = builder.create<arith::AndIOp>(loc, guardExpr, operands[j]);
        } else {
          guardExpr = builder.create<arith::OrIOp>(loc, guardExpr, operands[j]);
        }
      }

      if (!guardExpr) {
        dbg("  operand[" + llvm::Twine(i) + "] single-operand gate, walk unconditionally");
        ++gWalkDepth;
        walkValue(builder, operands[i], visited);
        --gWalkDepth;
        continue;
      }

      Value zero = builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(guardExpr.getType(), 0));
      arith::CmpIPredicate cmpPred =
          isAnd ? arith::CmpIPredicate::ne : arith::CmpIPredicate::eq;
      Value cond = builder.create<arith::CmpIOp>(loc, cmpPred, guardExpr, zero);

      dbg("  operand[" + llvm::Twine(i) + "] guard: " +
          (isAnd ? "AND(others)!=0" : "OR(others)==0") +
          " -> scf.if");
      auto ifOp = builder.create<scf::IfOp>(loc, cond, /*withElseRegion=*/false);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPoint(ifOp.thenBlock()->getTerminator());
        DenseSet<Value> branchVisited = visited;
        ++gWalkDepth;
        walkValue(builder, operands[i], branchVisited);
        --gWalkDepth;
      }
    }
    return;
  }

  // Generic combinational op: log the op name and recurse into all operands.
  dbg(defOp->getName().getStringRef() +
      "  " + llvm::Twine(defOp->getNumOperands()) + " operands -> recurse all");
  ++gWalkDepth;
  for (Value operand : defOp->getOperands())
    walkValue(builder, operand, visited);
  --gWalkDepth;
}

// =======================================================================
// Emit helpers
// =======================================================================

void EmitCausalityPass::emitBegin(OpBuilder &b, Location loc, int64_t sinkId,
                                   int64_t wsid, Value newVal,
                                   int32_t numBits) {
  b.create<func::CallOp>(
      loc, beginDecl,
      ValueRange{cI64(b, loc, sinkId), cI64(b, loc, wsid),
                 toI64(b, loc, newVal), cI32(b, loc, numBits)});
}

void EmitCausalityPass::emitAddPred(OpBuilder &b, Location loc, int64_t predId,
                                     int8_t role, int8_t delta) {
  b.create<func::CallOp>(
      loc, addPredDecl,
      ValueRange{cI64(b, loc, predId), cI8(b, loc, role), cI8(b, loc, delta)});
}

void EmitCausalityPass::emitCommit(OpBuilder &b, Location loc) {
  b.create<func::CallOp>(loc, commitDecl, ValueRange{});
}

// =======================================================================
// Value helpers
// =======================================================================

Value EmitCausalityPass::toI64(OpBuilder &b, Location loc, Value val) {
  auto i64 = b.getI64Type();
  if (val.getType() == i64)
    return val;
  if (auto itype = dyn_cast<IntegerType>(val.getType())) {
    if (itype.getWidth() < 64)
      return b.create<arith::ExtUIOp>(loc, i64, val);
    return b.create<arith::TruncIOp>(loc, i64, val);
  }
  return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(0));
}

Value EmitCausalityPass::cI64(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(v));
}

Value EmitCausalityPass::cI32(OpBuilder &b, Location loc, int32_t v) {
  return b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(v));
}

Value EmitCausalityPass::cI8(OpBuilder &b, Location loc, int8_t v) {
  return b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(b.getIntegerType(8), v));
}
