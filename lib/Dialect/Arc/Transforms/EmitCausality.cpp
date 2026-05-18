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
//   2. (Instrumentation) For each arc.state_write whose handle name is in
//      sinkNames, insert causality_begin / causality_add_pred / causality_commit
//      calls before the write.  Path-sensitivity is achieved by splitting on
//      any direct comb.mux feeding the written value.
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
// DFS: collect all arc.state_read storage handles reachable from val
// without crossing a state-write boundary.
// -----------------------------------------------------------------------
static void collectStateHandles(Value val, SmallVectorImpl<Value> &handles,
                                 DenseSet<Value> &visited) {
  if (!visited.insert(val).second)
    return;
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return;
  if (auto readOp = dyn_cast<StateReadOp>(defOp)) {
    handles.push_back(readOp.getState());
    return;
  }
  if (isa<StateWriteOp>(defOp))
    return;
  for (Value operand : defOp->getOperands())
    collectStateHandles(operand, handles, visited);
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

  llvm::StringSet<> sinkSet;
  {
    StringRef sr(opts.sinkNames);
    while (!sr.empty()) {
      auto [head, tail] = sr.split(',');
      head = head.trim();
      if (!head.empty())
        sinkSet.insert(head);
      sr = tail;
    }
  }

  SmallVector<StateWriteOp> writes;
  module.walk([&](StateWriteOp op) { writes.push_back(op); });

  for (auto writeOp : writes) {
    StringRef name = getStateName(writeOp.getState());
    if (name.empty() || !sinkSet.count(name))
      continue;
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

  // Resolve the write-site ID for this specific arc.state_write op.
  int64_t wsid = writeOpToSiteId.lookup(writeOp.getOperation());

  // Case A: direct comb.mux → path-sensitive scf.if
  if (auto muxOp = writtenVal.getDefiningOp<comb::MuxOp>()) {
    Value cond = muxOp.getCond();
    Value trueVal = muxOp.getTrueValue();
    Value falseVal = muxOp.getFalseValue();

    int64_t trueId = -1, falseId = -1, condId = -1;
    if (auto s = resolveDirectState(trueVal))
      trueId = lookupId(s);
    if (auto s = resolveDirectState(falseVal))
      falseId = lookupId(s);
    if (auto s = resolveDirectState(cond))
      condId = lookupId(s);

    // Insert scf.if before the state_write (both blocks get yield terminators
    // automatically when withElseRegion=true and no result types).
    auto ifOp =
        builder.create<scf::IfOp>(loc, cond, /*withElseRegion=*/true);

    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(ifOp.thenBlock()->getTerminator());
      emitBegin(builder, loc, sinkId, wsid, trueVal, numBits);
      if (trueId != -1) emitAddPred(builder, loc, trueId, ROLE_DATA, 0);
      if (condId != -1) emitAddPred(builder, loc, condId, ROLE_CONTROL_GUARD, 0);
      emitAddPred(builder, loc, sinkId, ROLE_PRIOR_STATE, -1);
      emitCommit(builder, loc);
    }
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(ifOp.elseBlock()->getTerminator());
      emitBegin(builder, loc, sinkId, wsid, falseVal, numBits);
      if (falseId != -1) emitAddPred(builder, loc, falseId, ROLE_DATA, 0);
      if (condId != -1) emitAddPred(builder, loc, condId, ROLE_CONTROL_GUARD, 0);
      emitAddPred(builder, loc, sinkId, ROLE_PRIOR_STATE, -1);
      emitCommit(builder, loc);
    }
    return;
  }

  // Case B: direct arc.state_read
  if (auto srcState = resolveDirectState(writtenVal)) {
    int64_t srcId = lookupId(srcState);
    emitBegin(builder, loc, sinkId, wsid, writtenVal, numBits);
    if (srcId != -1) emitAddPred(builder, loc, srcId, ROLE_DATA, 0);
    emitAddPred(builder, loc, sinkId, ROLE_PRIOR_STATE, -1);
    emitCommit(builder, loc);
    return;
  }

  // Case C: general combinational expression — DFS to collect all state reads
  SmallVector<Value> handles;
  DenseSet<Value> visited;
  collectStateHandles(writtenVal, handles, visited);

  emitBegin(builder, loc, sinkId, wsid, writtenVal, numBits);
  for (auto handle : handles) {
    int64_t predId = lookupId(handle);
    if (predId != -1)
      emitAddPred(builder, loc, predId, ROLE_DATA, 0);
  }
  emitAddPred(builder, loc, sinkId, ROLE_PRIOR_STATE, -1);
  emitCommit(builder, loc);
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
