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
//      causality_commit pair around a call to a memoized value recorder.
//      Each SSA Value in the cone gets ONE private func.func ("recorder") that
//      emits exactly the add_pred / add_comb_pred calls it contributes, with
//      scf.if guards preserved for path-sensitive branching.  Recorders are
//      shared across all callers: a reconvergent sub-expression is emitted once
//      rather than duplicated per path, collapsing O(paths) IR blowup to
//      O(nodes).
//
// Correctness argument for memoization:
//   - The predecessor contribution of a value V depends only on V's own
//     combinational cone, never on which path led to V.
//   - The runtime (causality_runtime.cc) deduplicates predecessors by
//     (pred_id, role, time_delta) and comb-preds by (comb_site_id, operand_idx)
//     at commit time, so call multiplicity is irrelevant to the result.
//   - Guarding (scf.if) is fully preserved in each recorder body, so only the
//     runtime-taken branch's state reads contribute predecessors.
//
// Registration: this pass is registered manually in arcilator.cpp (not via
// ArcPasses.td) so that touching ArcPasses.td and triggering a full rebuild
// of all Arc transforms is avoided.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/EmitCausalityPass.h"

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
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
// arcilator to get a per-write-site trace of every recorder built.
// ---------------------------------------------------------------------------
static bool causalityDebug() {
  static int cached = -1;
  if (cached == -1)
    cached = (::getenv("CAUSALITY_DEBUG") != nullptr) ? 1 : 0;
  return cached == 1;
}

static void dbg(const llvm::Twine &msg) {
  if (!causalityDebug()) return;
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
    // Tap-derived alloc_state (from --observe-named-values: AddTaps -> arc.tap ->
    // LowerState writes the combinational value to an alloc_state every cycle) stores its
    // name in a "names" ArrayAttr, not "name". Surface it so the tapped COMBINATIONAL
    // observable (e.g. io_trace_insns_0_insn) is referenceable as a sink by config.json /
    // the observable pool — its write events + cone are already instrumented like any state
    // write. Sink-id enumeration is by walk order (independent of names), so ids are stable.
    if (auto namesAttr = allocOp->getAttrOfType<ArrayAttr>("names"))
      for (Attribute a : namesAttr)
        if (auto s = dyn_cast<StringAttr>(a))
          if (!s.getValue().empty())
            return s.getValue();
    return {};
  }
  if (auto rootIn = dyn_cast<RootInputOp>(defOp))
    return rootIn.getName();
  if (auto rootOut = dyn_cast<RootOutputOp>(defOp))
    return rootOut.getName();
  return {};
}

// -----------------------------------------------------------------------
// Utility: get the "name" attribute of a memory handle's defining op
// (arc.memory / arc.alloc_memory), or empty. Used to scope + name memory
// cells (NEXT_STEPS #12).
// -----------------------------------------------------------------------
static StringRef getMemoryName(Value mem) {
  Operation *defOp = mem.getDefiningOp();
  if (!defOp)
    return {};
  if (auto nameAttr = defOp->getAttrOfType<StringAttr>("name"))
    return nameAttr.getValue();
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

  // comb.and / comb.or Operation* → 1-indexed comb site ID
  DenseMap<Operation *, int64_t> combOpToSiteId;

  // comb.mux ops carrying a trace.mux_site_id (forced mux-select, #5 v2). Separate
  // additive id space; the path-sensitive mux recorder emits a decision event per site.
  DenseMap<Operation *, int64_t> muxOpToSiteId;

  // NEXT_STEPS #12: memory-cell instrumentation. A memory is in scope if its
  // "name" attr contains any token from --causality-memories. Each in-scope
  // memory gets a contiguous sink-id range [base, base+depth) allocated ABOVE
  // the dense register id space (enumMemories). A cell's sink id = base + addr,
  // computed at runtime, so memory writes/reads reuse the register event +
  // time-delta machinery (no slicer change). Empty scope => no memory entries
  // => __signal_index.json + traces byte-identical to register-only.
  struct MemInfo {
    int64_t base;
    int64_t depth;
    std::string name;
  };
  SmallVector<std::string> memScopeTokens;
  DenseMap<Operation *, MemInfo> memToInfo; // memory defining-op -> info
  SmallVector<MemInfo> orderedMems;         // deterministic order, for the index

  // External function declarations added to the module
  func::FuncOp beginDecl, addPredDecl, commitDecl, addCombPredDecl,
      addMuxDecisionDecl, recorderOnceDecl;

  // -----------------------------------------------------------------------
  // Memoized recorder functions.
  //
  // Each SSA Value in the combinational cone gets ONE private func.func that
  // emits the add_pred / add_comb_pred calls that value contributes, with all
  // scf.if guards preserved.  Recorders are shared across every call site that
  // reaches that value (mux branches, multiple write sites), collapsing the
  // O(paths) IR blowup of the old inline-recursive approach to O(nodes).
  //
  // RecorderInfo holds:
  //   funcOp  — the emitted function.
  //   liveIns — the ordered set of model-body SSA values that must be passed
  //             as call arguments (mux conditions, and/or operands needed for
  //             runtime guards inside the recorder body).
  // -----------------------------------------------------------------------
  struct RecorderInfo {
    func::FuncOp funcOp;
    SmallVector<Value> liveIns;
  };
  DenseMap<Value, RecorderInfo> recorderCache;
  unsigned recorderCounter = 0;

  // Runtime-invocation memoization. Each recorder gets a stable integer id
  // (assigned lazily at its first emitRecorderCall). At runtime, a per-event
  // guard (causality_recorder_once) lets each recorder body run AT MOST ONCE
  // per commit — so a reconvergent sub-cone shared by many callers executes
  // once (O(nodes)) instead of once-per-path (O(paths)). Byte-identical output
  // (a value's predecessor contribution is path-independent).
  DenseMap<Operation *, int64_t> recIdMap;
  int64_t recIdCounter = 0;

  // Emit or retrieve the memoized recorder for a given SSA value.
  RecorderInfo getOrEmitRecorder(Value val);

  // Emit a func.call to a recorder, mapping its liveIns through argMap
  // (model-SSA-value → block-arg of the enclosing recorder body).
  void emitRecorderCall(OpBuilder &b, Location loc, const RecorderInfo &info,
                        const DenseMap<Value, Value> &argMap);

  void enumSignals();
  void enumMemories(); // NEXT_STEPS #12
  void enumWriteSites();
  void enumCombSites();
  void enumMuxSites();
  void writeSignalIndex();
  void declareRuntimeFuncs();
  void injectForWrite(OpBuilder &b, StateWriteOp writeOp, int64_t sinkId);
  void injectForMemWrite(OpBuilder &b, MemoryWriteOp writeOp, const MemInfo &mi,
                         int64_t wsid); // NEXT_STEPS #12

  void emitBegin(OpBuilder &b, Location loc, int64_t sinkId, int64_t wsid,
                 Value newVal, int32_t numBits);
  void emitAddPred(OpBuilder &b, Location loc, int64_t predId, int8_t role,
                   int8_t delta);
  void emitAddCombPred(OpBuilder &b, Location loc, int64_t siteId,
                       int64_t operand, Value isIdentityI8);
  void emitAddMuxDecision(OpBuilder &b, Location loc, int64_t siteId,
                          Value selValueI8, Value inputsDifferI8);
  void emitCommit(OpBuilder &b, Location loc);

  // NEXT_STEPS #12: dynamic (runtime-Value) sink/pred ids for memory cells.
  void emitBeginDynamic(OpBuilder &b, Location loc, Value sinkI64, int64_t wsid,
                        Value newVal, int32_t numBits);
  void emitAddPredDynamic(OpBuilder &b, Location loc, Value predI64, int8_t role,
                          int8_t delta);
  // cell sink id = base + zext(addr), computed in MLIR.
  Value cellId(OpBuilder &b, Location loc, int64_t base, Value addr);

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

  // Reset memoization state (pass instance may be reused across modules).
  recorderCache.clear();
  recorderCounter = 0;
  recIdMap.clear();
  recIdCounter = 0;

  enumSignals();
  enumMemories(); // #12: must follow enumSignals (allocates above reg watermark)
  enumCombSites();
  enumMuxSites();
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

  // NEXT_STEPS #12: instrument in-scope memory writes as cell write events.
  if (!memToInfo.empty()) {
    SmallVector<MemoryWriteOp> memWrites;
    module.walk([&](MemoryWriteOp op) { memWrites.push_back(op); });
    for (auto writeOp : memWrites) {
      Operation *md = writeOp.getMemory().getDefiningOp();
      auto it = memToInfo.find(md);
      if (it == memToInfo.end())
        continue;
      int64_t wsid = writeOpToSiteId.lookup(writeOp.getOperation());
      OpBuilder builder(writeOp);
      injectForMemWrite(builder, writeOp, it->second, wsid);
    }
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
  // #12: in-scope memory writes continue the write-site id space AFTER the
  // register writes, so register write-site ids are unchanged when off.
  if (!memToInfo.empty()) {
    module.walk([&](MemoryWriteOp op) {
      if (memToInfo.count(op.getMemory().getDefiningOp()))
        writeOpToSiteId[op.getOperation()] = counter++;
    });
  }
}

// =======================================================================
// Phase 1d (NEXT_STEPS #12): enumerate in-scope memories; allocate a
// contiguous sink-id range [base, base+depth) per memory ABOVE the register
// watermark.  No-op when --causality-memories is empty (byte-identical).
// =======================================================================
void EmitCausalityPass::enumMemories() {
  memScopeTokens.clear();
  for (StringRef s(opts.memoryNames); !s.empty();) {
    auto [tok, rest] = s.split(',');
    tok = tok.trim();
    if (!tok.empty())
      memScopeTokens.push_back(tok.str());
    s = rest;
  }
  if (memScopeTokens.empty())
    return; // feature OFF.

  // Cell ids start above the dense register id space (1..N).
  int64_t nextBase = static_cast<int64_t>(storageToId.size()) + 1;

  // Collect distinct memories from their read/write ops, in deterministic walk
  // order (robust to whether the handle is arc.memory or arc.alloc_memory).
  SmallVector<Value> memVals;
  llvm::DenseSet<Operation *> seen;
  module.walk([&](Operation *op) {
    Value mem;
    if (auto w = dyn_cast<MemoryWriteOp>(op))
      mem = w.getMemory();
    else if (auto r = dyn_cast<MemoryReadOp>(op))
      mem = r.getMemory();
    if (!mem)
      return;
    Operation *md = mem.getDefiningOp();
    if (md && seen.insert(md).second)
      memVals.push_back(mem);
  });

  llvm::StringSet<> matchedTokens;
  for (Value mem : memVals) {
    StringRef name = getMemoryName(mem);
    llvm::errs() << "[causality] #12 memory candidate: '"
                 << (name.empty() ? StringRef("<unnamed>") : name) << "'\n";
    if (name.empty())
      continue;
    StringRef matched;
    for (auto &tok : memScopeTokens)
      if (name.contains(tok)) {
        matched = tok;
        matchedTokens.insert(tok);
        break;
      }
    if (matched.empty())
      continue;
    auto memTy = dyn_cast<MemoryType>(mem.getType());
    if (!memTy)
      continue;
    int64_t depth = static_cast<int64_t>(memTy.getNumWords());
    // Stage B (trace format v4) serializes sink_id/pred_id as int64, so cell-id ranges
    // are bounded only by INT64_MAX (Stage A's int32 guard is lifted). The fail-loud
    // remains for the pathological/overflow case — a range past INT64_MAX means a bug
    // upstream, never a real memory.
    if (nextBase < 0 || depth < 0 || nextBase > 0x7fffffffffffffffLL - depth) {
      module.emitError("EmitCausality #12: memory '")
          << name << "' cell-id range [" << nextBase << ", " << nextBase + depth
          << ") overflows int64 — upstream bug (memory depth or base is implausible).";
      signalPassFailure();
      return;
    }
    MemInfo mi{nextBase, depth, name.str()};
    memToInfo[mem.getDefiningOp()] = mi;
    orderedMems.push_back(mi);
    llvm::errs() << "[causality] #12 memory IN SCOPE: '" << name << "' base="
                 << mi.base << " depth=" << depth << "\n";
    nextBase += depth;
  }

  // Fail loud if a configured token matched no memory (no silent no-op).
  for (auto &tok : memScopeTokens)
    if (!matchedTokens.contains(tok)) {
      module.emitError("EmitCausality #12: --causality-memories token '")
          << tok << "' matched no memory (see the memory-candidate list above).";
      signalPassFailure();
      return;
    }
}

// =======================================================================
// Phase 1c: enumerate comb.and / comb.or ops and assign comb site IDs
// =======================================================================

void EmitCausalityPass::enumCombSites() {
  // Source-anchored comb-site IDs: each comb.and/comb.or carries a stable
  // `trace.comb_site_id` attribute assigned at the top.mlir level (before
  // arc lowering / inlining) by the comb-site tagging build step.  We READ that
  // id rather than re-counting, so the id names the same SOURCE gate in the
  // causality trace, the simulated fault, and the ExportVerilog miter — even
  // though inlining duplicates a source gate into many arc-level copies (all
  // copies inherit the same id, and the runtime de-dups comb-preds by
  // (comb_site_id, operand_idx), so the per-source view falls out for free).
  //
  // Comb ops WITHOUT the attribute were synthesized during lowering and have no
  // source counterpart; they get no id (lookup returns 0) and are skipped as
  // injection candidates, while the path-sensitive walk still descends through
  // them (see getOrEmitRecorder).
  module.walk([&](Operation *op) {
    if (!isa<comb::AndOp, comb::OrOp>(op))
      return;
    if (auto a = op->getAttrOfType<IntegerAttr>("trace.comb_site_id"))
      combOpToSiteId[op] = a.getInt();
  });
}

// Collect the source mux ids encoded in a synthetic location (#5 v2): a
// FileLineColLoc with filename "mux_site" carries the source id in its line field
// (stamped by tools/tag_comb_sites.py tag-mux). Recurses FusedLoc (a CombFolds blend
// of two source muxes produces a fused loc) and NameLoc. Returns the SET of distinct ids.
static void collectMuxSiteIds(Location loc, llvm::SmallDenseSet<int64_t, 2> &ids) {
  if (auto flc = dyn_cast<FileLineColLoc>(loc)) {
    if (flc.getFilename().getValue() == "mux_site")
      ids.insert(static_cast<int64_t>(flc.getLine()));
  } else if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location sub : fused.getLocations())
      collectMuxSiteIds(sub, ids);
  } else if (auto named = dyn_cast<NameLoc>(loc)) {
    collectMuxSiteIds(named.getChildLoc(), ids);
  }
}

void EmitCausalityPass::enumMuxSites() {
  // Source-anchored mux-site IDs via SYNTHETIC LOCATION (#5 v2 forced-mux-select).
  // tag-mux stamps each source comb.mux with loc("mux_site":N:0). Arc-lowering rewrites
  // (canonicalization, inlining-context rebuilds) DROP custom attributes but PRESERVE
  // source locations, so the cone muxes the recorder walks carry their source id in
  // op->getLoc() even after the trace.mux_site_id ATTR is gone. (That attr-vs-loc
  // difference is exactly why the earlier attr-reading recorder recorded 0.)
  //
  // Validated on rocket (synthetic-loc diagnostic, lowered to here): of 2346 cone muxes,
  // 99.1% carry an exact single-source mux_site loc, 0% are fused/blended, 0.9% are
  // lowering-synthesized (register-reset / memory-port muxes — no source conditional).
  //
  // Per mux, read the mux_site id(s) from the location:
  //   exactly 1 id  -> the source mux id (the clean, common case): record under it.
  //   >1 id (fused) -> a CombFolds blend merged two source conditionals; the recorded
  //                    trigger can't be cleanly attributed to ONE -> EXCLUDE (id 0, not
  //                    selectable). Counted + reported — never silently mis-attributed.
  //   0 ids         -> lowering-synthesized (reg reset / mem port) -> skip (no source).
  int64_t nClean = 0, nFused = 0, nSynth = 0;
  module.walk([&](Operation *op) {
    if (!isa<comb::MuxOp>(op))
      return;
    llvm::SmallDenseSet<int64_t, 2> ids;
    collectMuxSiteIds(op->getLoc(), ids);
    if (ids.size() == 1) {
      muxOpToSiteId[op] = *ids.begin();
      ++nClean;
    } else if (ids.size() > 1) {
      ++nFused; // blended: excluded (lookup returns 0 -> recorder treats as untagged)
    } else {
      ++nSynth; // synthesized: no source conditional
    }
  });
  dbg("enumMuxSites (source-loc): " + llvm::Twine(nClean) + " clean / " +
      llvm::Twine(nFused) + " fused-excluded / " + llvm::Twine(nSynth) +
      " synthesized");
}

// =======================================================================
// Phase 2: write __signal_index.json
// =======================================================================

void EmitCausalityPass::writeSignalIndex() {
  llvm::json::Array signals;
  int64_t counter = 1;

  module.walk([&](Operation *op) {
    Value stateResult;
    if (auto v = dyn_cast<RootInputOp>(op))
      stateResult = v.getState();
    else if (auto v = dyn_cast<RootOutputOp>(op))
      stateResult = v.getState();
    else if (auto v = dyn_cast<AllocStateOp>(op))
      stateResult = v.getState();
    if (!stateResult)
      return;

    // Resolve the signal name via getStateName (NOT the singular "name" attr
    // only). Tap-derived alloc_states (from --observe-named-values: AddTaps ->
    // arc.tap -> LowerState writes the COMBINATIONAL value to an alloc_state
    // every cycle) store their name in a "names" ArrayAttr, not "name", so the
    // old singular-"name" read left them as "reg_<id>" in this `signals` array —
    // the exact table the observable resolver / slice gate looks up by name.
    // That is why combinational observation-sinks like BOOM
    // io_trace_insns_0_{iaddr,insn,wdata} (present in `write_sites` via
    // getStateName, but "reg_N" here) failed to resolve. getStateName surfaces
    // them by their real name. IDs are unchanged (walk order, name-independent),
    // so this is additive: already-named states keep their id; previously
    // unnamed tap states gain their real name. A build with NO taps (e.g.
    // Rocket, no --observe-named-values) has no "names" attrs => byte-identical.
    StringRef name = getStateName(stateResult);

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
  // #12: append in-scope memory write-sites (identified by memory_name; the
  // target cell is dynamic so there is no single signal_id).
  if (!memToInfo.empty()) {
    module.walk([&](MemoryWriteOp op) {
      auto it = memToInfo.find(op.getMemory().getDefiningOp());
      if (it == memToInfo.end())
        return;
      int64_t wsid = writeOpToSiteId.lookup(op.getOperation());
      if (wsid == 0)
        return;
      llvm::json::Object e;
      e["write_site_id"] = wsid;
      e["memory_name"] = it->second.name;
      writeSites.push_back(llvm::json::Value(std::move(e)));
    });
  }
  root["write_sites"] = llvm::json::Value(std::move(writeSites));

  // Comb site table: one entry per SOURCE comb.and / comb.or (keyed by the
  // source-anchored trace.comb_site_id).  Inlining duplicates a source gate into
  // many arc-level copies sharing one id; we emit a single entry per id (the
  // first copy seen).  Untagged synthesized gates (id 0) are not injection
  // candidates and are skipped.
  llvm::json::Array combSites;
  llvm::DenseSet<int64_t> seenCombIds;
  module.walk([&](Operation *op) {
    if (!isa<comb::AndOp, comb::OrOp>(op))
      return;
    int64_t csid = combOpToSiteId.lookup(op);
    if (csid == 0)
      return;
    if (!seenCombIds.insert(csid).second)
      return;
    llvm::json::Object e;
    e["comb_site_id"] = csid;
    e["gate"] = isa<comb::AndOp>(op) ? "and" : "or";
    e["num_operands"] = static_cast<int64_t>(op->getNumOperands());
    combSites.push_back(llvm::json::Value(std::move(e)));
  });
  root["comb_sites"] = llvm::json::Value(std::move(combSites));

  // Mux site table (#5 v2): one entry per comb.mux (keyed by the pass-assigned
  // mux_site_id from enumMuxSites; deduped if the same op is walked twice).
  llvm::json::Array muxSites;
  llvm::DenseSet<int64_t> seenMuxIds;
  module.walk([&](Operation *op) {
    if (!isa<comb::MuxOp>(op))
      return;
    int64_t msid = muxOpToSiteId.lookup(op);
    if (msid == 0)
      return;
    if (!seenMuxIds.insert(msid).second)
      return;
    llvm::json::Object e;
    e["mux_site_id"] = msid;
    muxSites.push_back(llvm::json::Value(std::move(e)));
  });
  root["mux_sites"] = llvm::json::Value(std::move(muxSites));

  // #12: memory table (additive; only emitted when memories are instrumented).
  // Each memory owns cell sink ids [base_id, base_id+depth).  Consumers expand a
  // memory observable name to that range.  Absent when the feature is off ⇒
  // __signal_index.json byte-identical to register-only.
  if (!orderedMems.empty()) {
    llvm::json::Array mems;
    for (auto &mi : orderedMems) {
      llvm::json::Object e;
      e["name"] = mi.name;
      e["base_id"] = mi.base;
      e["depth"] = mi.depth;
      mems.push_back(llvm::json::Value(std::move(e)));
    }
    root["memories"] = llvm::json::Value(std::move(mems));
  }

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
  addCombPredDecl = getOrDeclare("causality_add_comb_pred",
                                  FunctionType::get(ctx, {i64, i64, i8}, {}));
  addMuxDecisionDecl = getOrDeclare("causality_add_mux_decision",
                                     FunctionType::get(ctx, {i64, i8, i8}, {}));
  // Runtime-invocation memoization guard: (recorder_id) -> 1 if first time this
  // event, 0 if already run. i32 result (unambiguous ABI vs C int32_t).
  recorderOnceDecl = getOrDeclare("causality_recorder_once",
                                   FunctionType::get(ctx, {i64}, {i32}));
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

  // Get or build the shared recorder for writtenVal, then call it.
  // The recorder's liveIns are model SSA values directly available here.
  RecorderInfo info = getOrEmitRecorder(writtenVal);
  builder.create<func::CallOp>(loc, info.funcOp,
                                ValueRange(info.liveIns));

  emitCommit(builder, loc);
}

// NEXT_STEPS #12: instrument an in-scope arc.memory_write as a cell write event.
// sink_id = base + addr (runtime), value = the written word; the recorder walks
// the written data's cone exactly like a register write.
void EmitCausalityPass::injectForMemWrite(OpBuilder &builder,
                                          MemoryWriteOp writeOp,
                                          const MemInfo &mi, int64_t wsid) {
  Location loc = writeOp.getLoc();
  Value addr = writeOp.getAddress();
  Value data = writeOp.getData();
  int32_t numBits = 0;
  if (auto itype = dyn_cast<IntegerType>(data.getType()))
    numBits = static_cast<int32_t>(itype.getWidth());

  Value sinkVal = cellId(builder, loc, mi.base, addr);
  emitBeginDynamic(builder, loc, sinkVal, wsid, data, numBits);
  RecorderInfo info = getOrEmitRecorder(data);
  builder.create<func::CallOp>(loc, info.funcOp, ValueRange(info.liveIns));
  emitCommit(builder, loc);
}

// =======================================================================
// getOrEmitRecorder: memoized per-value recorder emission
//
// For each distinct SSA value in the combinational cone, emit ONE private
// func.func at module level and cache it.  The function body mirrors the
// original walkValue logic but uses calls to child recorders instead of
// inline re-walks, and scf.if guards are preserved for path-sensitivity.
//
// Live-in collection:
//   The recorder body needs certain model SSA values at runtime (mux conds
//   and and/or operands, to evaluate guards).  These become function args.
//   The ordered set is computed while building each recorder by unioning the
//   values it directly references with the liveIns of every child recorder.
// =======================================================================

EmitCausalityPass::RecorderInfo
EmitCausalityPass::getOrEmitRecorder(Value val) {
  // Return cached recorder if already built.
  auto it = recorderCache.find(val);
  if (it != recorderCache.end())
    return it->second;

  Location loc = val.getLoc();
  Operation *defOp = val.getDefiningOp();

  // -----------------------------------------------------------------------
  // Boundary: block arg (no defOp), StateWriteOp boundary, or zero-operand
  // op (e.g. hw.constant).  No predecessor state reads reachable → empty
  // recorder.
  // -----------------------------------------------------------------------
  if (!defOp || isa<StateWriteOp>(defOp) || defOp->getNumOperands() == 0) {
    auto funcType = FunctionType::get(ctx, {}, {});
    OpBuilder b(ctx);
    b.setInsertionPointToEnd(module.getBody());
    std::string name = "__caus_rec_" + std::to_string(recorderCounter++);
    dbg("boundary -> empty recorder " + name);
    auto recFunc = b.create<func::FuncOp>(loc, name, funcType);
    recFunc.setVisibility(SymbolTable::Visibility::Private);
    Block *body = recFunc.addEntryBlock();
    OpBuilder bodyBuilder(body, body->end());
    bodyBuilder.create<func::ReturnOp>(loc);
    RecorderInfo info{recFunc, {}};
    recorderCache[val] = info;
    return info;
  }

  // -----------------------------------------------------------------------
  // arc.state_read leaf: emit a single DATA add_pred call.
  // -----------------------------------------------------------------------
  if (auto readOp = dyn_cast<StateReadOp>(defOp)) {
    int64_t predId = lookupId(readOp.getState());
    StringRef name = getStateName(readOp.getState());
    dbg("state_read id=" + llvm::Twine(predId) +
        " (" + (name.empty() ? "?" : name) + ") -> recorder with add_pred(DATA)");
    auto funcType = FunctionType::get(ctx, {}, {});
    OpBuilder b(ctx);
    b.setInsertionPointToEnd(module.getBody());
    std::string recName = "__caus_rec_" + std::to_string(recorderCounter++);
    auto recFunc = b.create<func::FuncOp>(loc, recName, funcType);
    recFunc.setVisibility(SymbolTable::Visibility::Private);
    Block *body = recFunc.addEntryBlock();
    OpBuilder bodyBuilder(body, body->end());
    if (predId != -1)
      emitAddPred(bodyBuilder, loc, predId, ROLE_DATA, -1);
    bodyBuilder.create<func::ReturnOp>(loc);
    RecorderInfo info{recFunc, {}};
    recorderCache[val] = info;
    return info;
  }

  // -----------------------------------------------------------------------
  // arc.memory_read (NEXT_STEPS #12): an in-scope memory read links to the cell
  // it read (cell_sink_id = base + addr) via a DATA predecessor — the existing
  // time-delta=-1 resolution makes that "the last writer of that cell" — and
  // recurses the ADDRESS cone so guards that select the address enter the slice.
  // Out-of-scope memories fall through to the generic handler (address-only, as
  // before), so the feature-off behaviour is byte-identical.
  // -----------------------------------------------------------------------
  if (auto readOp = dyn_cast<MemoryReadOp>(defOp)) {
    auto mi = memToInfo.find(readOp.getMemory().getDefiningOp());
    if (mi != memToInfo.end()) {
      Value addr = readOp.getAddress();
      RecorderInfo addrInfo = getOrEmitRecorder(addr);

      llvm::SetVector<Value> liveInSet;
      liveInSet.insert(addr);
      for (auto v : addrInfo.liveIns)
        liveInSet.insert(v);
      SmallVector<Value> liveIns(liveInSet.begin(), liveInSet.end());

      SmallVector<Type> argTypes;
      for (auto v : liveIns)
        argTypes.push_back(v.getType());
      auto funcType = FunctionType::get(ctx, argTypes, {});

      OpBuilder modBuilder(ctx);
      modBuilder.setInsertionPointToEnd(module.getBody());
      std::string recName = "__caus_rec_" + std::to_string(recorderCounter++);
      dbg("memory_read in-scope base=" + llvm::Twine(mi->second.base) +
          " -> recorder " + recName);
      auto recFunc = modBuilder.create<func::FuncOp>(loc, recName, funcType);
      recFunc.setVisibility(SymbolTable::Visibility::Private);

      RecorderInfo info{recFunc, liveIns};
      recorderCache[val] = info;

      Block *body = recFunc.addEntryBlock();
      OpBuilder bodyBuilder(body, body->end());
      DenseMap<Value, Value> argMap;
      for (auto [modelVal, blockArg] : llvm::zip(liveIns, body->getArguments()))
        argMap[modelVal] = blockArg;

      // cell sink id = base + zext(addr); link to the last writer of that cell.
      Value cell = cellId(bodyBuilder, loc, mi->second.base, argMap[addr]);
      emitAddPredDynamic(bodyBuilder, loc, cell, ROLE_DATA, -1);
      // recurse the address cone (guards selecting which cell is read).
      emitRecorderCall(bodyBuilder, loc, addrInfo, argMap);
      bodyBuilder.create<func::ReturnOp>(loc);
      return info;
    }
    // out of scope: fall through to the generic handler below.
  }

  // -----------------------------------------------------------------------
  // comb.mux: path-sensitive split.
  //
  // Recorder body:
  //   call rec_cond(...)          -- DATA predecessors of the condition
  //   [add_pred(cid, CONTROL_GUARD)] -- if cond is a direct state read
  //   if (cond) { call rec_true(...) } else { call rec_false(...) }
  //
  // Live-ins: {cond} ∪ liveIns(rec_cond) ∪ liveIns(rec_true) ∪ liveIns(rec_false)
  // -----------------------------------------------------------------------
  if (auto muxOp = dyn_cast<comb::MuxOp>(defOp)) {
    Value cond     = muxOp.getCond();
    Value trueVal  = muxOp.getTrueValue();
    Value falseVal = muxOp.getFalseValue();
    // #5 v2: tagged mux -> also record a decision event (sel_value, inputs_differ).
    int64_t muxSiteId = muxOpToSiteId.lookup(defOp);

    dbg("comb.mux -> build mux recorder");
    RecorderInfo condInfo  = getOrEmitRecorder(cond);
    RecorderInfo trueInfo  = getOrEmitRecorder(trueVal);
    RecorderInfo falseInfo = getOrEmitRecorder(falseVal);

    llvm::SetVector<Value> liveInSet;
    liveInSet.insert(cond);
    // The two data inputs are ALWAYS needed: inputs_differ = (true != false) gates both
    // the #5 decision event AND the precise mux-predecessor slice (steps 1-2 below).
    liveInSet.insert(trueVal);
    liveInSet.insert(falseVal);
    for (auto v : condInfo.liveIns)  liveInSet.insert(v);
    for (auto v : trueInfo.liveIns)  liveInSet.insert(v);
    for (auto v : falseInfo.liveIns) liveInSet.insert(v);
    SmallVector<Value> liveIns(liveInSet.begin(), liveInSet.end());

    SmallVector<Type> argTypes;
    for (auto v : liveIns) argTypes.push_back(v.getType());
    auto funcType = FunctionType::get(ctx, argTypes, {});

    OpBuilder modBuilder(ctx);
    modBuilder.setInsertionPointToEnd(module.getBody());
    std::string recName = "__caus_rec_" + std::to_string(recorderCounter++);
    auto recFunc = modBuilder.create<func::FuncOp>(loc, recName, funcType);
    recFunc.setVisibility(SymbolTable::Visibility::Private);

    // Cache before building body (acyclic, but good hygiene).
    RecorderInfo info{recFunc, liveIns};
    recorderCache[val] = info;

    Block *body = recFunc.addEntryBlock();
    OpBuilder bodyBuilder(body, body->end());

    // Map model SSA values → block args of this recorder.
    DenseMap<Value, Value> argMap;
    for (auto [modelVal, blockArg] :
         llvm::zip(liveIns, body->getArguments()))
      argMap[modelVal] = blockArg;

    Value mCond  = argMap[cond];
    Value mTrue  = argMap[trueVal];
    Value mFalse = argMap[falseVal];

    // inputs_differ = (trueVal != falseVal). The select — and therefore the select's
    // own cone (step 1) and its CONTROL_GUARD (step 2) — only CAUSES the mux output when
    // the two arms differ; when they are equal the output is independent of the select.
    // So this is the precise mux-predecessor condition (the mux analog of guard-removal's
    // is_identity), and the SAME value feeds the #5 decision event.
    Value diff = bodyBuilder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, mTrue, mFalse);

    // 0. Decision event for a tagged mux (#5 v2 broken-conditional): record the select
    //    value and inputs_differ — the only case where forcing the select changes the
    //    output. Unconditional: the call to this recorder is already under whatever
    //    conditions reach the mux.
    if (muxSiteId != 0) {
      Value selI8 = bodyBuilder.create<arith::ExtUIOp>(
          loc, bodyBuilder.getI8Type(), mCond);
      Value diffI8 = bodyBuilder.create<arith::ExtUIOp>(
          loc, bodyBuilder.getI8Type(), diff);
      emitAddMuxDecision(bodyBuilder, loc, muxSiteId, selI8, diffI8);
    }

    // 1+2. The select's DATA predecessors (its cone) and its CONTROL_GUARD are
    //      predecessors of the mux output ONLY when the arms differ — gate on
    //      inputs_differ (precise; was previously recorded unconditionally).
    {
      auto diffIf =
          bodyBuilder.create<scf::IfOp>(loc, diff, /*withElseRegion=*/false);
      OpBuilder::InsertionGuard g(bodyBuilder);
      bodyBuilder.setInsertionPoint(diffIf.thenBlock()->getTerminator());
      // 1. Walk the condition (DATA predecessors).
      emitRecorderCall(bodyBuilder, loc, condInfo, argMap);
      // 2. CONTROL_GUARD if condition is a direct state read.
      if (auto s = resolveDirectState(cond)) {
        int64_t cid = lookupId(s);
        if (cid != -1)
          emitAddPred(bodyBuilder, loc, cid, ROLE_CONTROL_GUARD, -1);
      }
    }

    // 3. Branch on runtime cond: only the taken arm contributes DATA preds.
    Value mappedCond = argMap[cond];
    auto ifOp =
        bodyBuilder.create<scf::IfOp>(loc, mappedCond, /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard g(bodyBuilder);
      bodyBuilder.setInsertionPoint(ifOp.thenBlock()->getTerminator());
      emitRecorderCall(bodyBuilder, loc, trueInfo, argMap);
    }
    {
      OpBuilder::InsertionGuard g(bodyBuilder);
      bodyBuilder.setInsertionPoint(ifOp.elseBlock()->getTerminator());
      emitRecorderCall(bodyBuilder, loc, falseInfo, argMap);
    }

    bodyBuilder.create<func::ReturnOp>(loc);
    return info;
  }

  // -----------------------------------------------------------------------
  // comb.and / comb.or: short-circuit masking guards.
  //
  // For each operand i, the guard is AND/OR of the other operands:
  //   AND: operand i matters only when AND(others) != 0
  //   OR:  operand i matters only when OR(others) == 0
  //
  // Recorder body (per operand i):
  //   guardExpr = AND/OR of mapped operands[j!=i]
  //   if single operand: add_comb_pred + call rec_operand_i (unconditional)
  //   otherwise:         if (guardExpr) { add_comb_pred; call rec_operand_i }
  //
  // Live-ins: all operands (for guard computation) ∪ child live-ins
  // -----------------------------------------------------------------------
  if (isa<comb::AndOp, comb::OrOp>(defOp)) {
    const bool isAnd = isa<comb::AndOp>(defOp);
    OperandRange operands = defOp->getOperands();
    int64_t siteId = combOpToSiteId.lookup(defOp);

    dbg(llvm::Twine(isAnd ? "comb.and" : "comb.or") +
        " " + llvm::Twine(operands.size()) + " operands -> build and/or recorder");

    SmallVector<RecorderInfo> operandInfos;
    for (auto operand : operands)
      operandInfos.push_back(getOrEmitRecorder(operand));

    llvm::SetVector<Value> liveInSet;
    for (auto operand : operands)
      liveInSet.insert(operand);
    for (auto &oi : operandInfos)
      for (auto v : oi.liveIns)
        liveInSet.insert(v);
    SmallVector<Value> liveIns(liveInSet.begin(), liveInSet.end());

    SmallVector<Type> argTypes;
    for (auto v : liveIns) argTypes.push_back(v.getType());
    auto funcType = FunctionType::get(ctx, argTypes, {});

    OpBuilder modBuilder(ctx);
    modBuilder.setInsertionPointToEnd(module.getBody());
    std::string recName = "__caus_rec_" + std::to_string(recorderCounter++);
    auto recFunc = modBuilder.create<func::FuncOp>(loc, recName, funcType);
    recFunc.setVisibility(SymbolTable::Visibility::Private);

    RecorderInfo info{recFunc, liveIns};
    recorderCache[val] = info;

    Block *body = recFunc.addEntryBlock();
    OpBuilder bodyBuilder(body, body->end());

    DenseMap<Value, Value> argMap;
    for (auto [modelVal, blockArg] :
         llvm::zip(liveIns, body->getArguments()))
      argMap[modelVal] = blockArg;

    SmallVector<Value> mappedOperands;
    for (auto operand : operands)
      mappedOperands.push_back(argMap[operand]);

    for (unsigned i = 0; i < operands.size(); ++i) {
      // Build guardExpr = AND/OR of all operands except i.
      Value guardExpr;
      for (unsigned j = 0; j < operands.size(); ++j) {
        if (j == i)
          continue;
        if (!guardExpr) {
          guardExpr = mappedOperands[j];
        } else if (isAnd) {
          guardExpr =
              bodyBuilder.create<arith::AndIOp>(loc, guardExpr, mappedOperands[j]);
        } else {
          guardExpr =
              bodyBuilder.create<arith::OrIOp>(loc, guardExpr, mappedOperands[j]);
        }
      }

      // is_identity: operands[i] == identity element (all-ones for AND, 0 for OR).
      auto computeIsIdentity = [&]() -> Value {
        unsigned W =
            cast<IntegerType>(mappedOperands[i].getType()).getWidth();
        APInt identVal = isAnd ? APInt::getAllOnes(W) : APInt(W, 0);
        Value identConst = bodyBuilder.create<arith::ConstantOp>(
            loc,
            bodyBuilder.getIntegerAttr(mappedOperands[i].getType(), identVal));
        Value cmpI1 = bodyBuilder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, mappedOperands[i], identConst);
        return bodyBuilder.create<arith::ExtUIOp>(loc, bodyBuilder.getI8Type(),
                                                   cmpI1);
      };

      if (!guardExpr) {
        // Single-operand gate: always propagates.  Record the comb-pred only
        // for tagged (source-anchored) gates; untagged synthesized gates still
        // have their operand walked, just not recorded as injection candidates.
        if (siteId != 0)
          emitAddCombPred(bodyBuilder, loc, siteId, static_cast<int64_t>(i),
                          computeIsIdentity());
        emitRecorderCall(bodyBuilder, loc, operandInfos[i], argMap);
        continue;
      }

      // Multi-operand: guard the walk.
      Value zero = bodyBuilder.create<arith::ConstantOp>(
          loc, bodyBuilder.getIntegerAttr(guardExpr.getType(), 0));
      arith::CmpIPredicate cmpPred =
          isAnd ? arith::CmpIPredicate::ne : arith::CmpIPredicate::eq;
      Value guardCond =
          bodyBuilder.create<arith::CmpIOp>(loc, cmpPred, guardExpr, zero);

      auto ifOp = bodyBuilder.create<scf::IfOp>(loc, guardCond,
                                                  /*withElseRegion=*/false);
      {
        OpBuilder::InsertionGuard g(bodyBuilder);
        bodyBuilder.setInsertionPoint(ifOp.thenBlock()->getTerminator());
        if (siteId != 0)
          emitAddCombPred(bodyBuilder, loc, siteId, static_cast<int64_t>(i),
                          computeIsIdentity());
        emitRecorderCall(bodyBuilder, loc, operandInfos[i], argMap);
      }
    }

    bodyBuilder.create<func::ReturnOp>(loc);
    return info;
  }

  // -----------------------------------------------------------------------
  // Generic combinational op: no runtime guards, just recurse all operands.
  // Live-ins: union of child live-ins only (operands not needed directly).
  // -----------------------------------------------------------------------
  SmallVector<RecorderInfo> operandInfos;
  for (auto operand : defOp->getOperands())
    operandInfos.push_back(getOrEmitRecorder(operand));

  {
    llvm::SetVector<Value> liveInSet;
    for (auto &oi : operandInfos)
      for (auto v : oi.liveIns)
        liveInSet.insert(v);
    SmallVector<Value> liveIns(liveInSet.begin(), liveInSet.end());

    SmallVector<Type> argTypes;
    for (auto v : liveIns) argTypes.push_back(v.getType());
    auto funcType = FunctionType::get(ctx, argTypes, {});

    OpBuilder modBuilder(ctx);
    modBuilder.setInsertionPointToEnd(module.getBody());
    std::string recName = "__caus_rec_" + std::to_string(recorderCounter++);
    dbg("generic op " + defOp->getName().getStringRef() +
        " -> recorder " + recName);
    auto recFunc = modBuilder.create<func::FuncOp>(loc, recName, funcType);
    recFunc.setVisibility(SymbolTable::Visibility::Private);

    RecorderInfo info{recFunc, liveIns};
    recorderCache[val] = info;

    Block *body = recFunc.addEntryBlock();
    OpBuilder bodyBuilder(body, body->end());

    DenseMap<Value, Value> argMap;
    for (auto [modelVal, blockArg] :
         llvm::zip(liveIns, body->getArguments()))
      argMap[modelVal] = blockArg;

    for (auto &oi : operandInfos)
      emitRecorderCall(bodyBuilder, loc, oi, argMap);

    bodyBuilder.create<func::ReturnOp>(loc);
    return info;
  }
}

// -----------------------------------------------------------------------
// emitRecorderCall: emit a func.call to a recorder, mapping its liveIns
// through argMap (model-SSA → enclosing recorder's block arg).
// -----------------------------------------------------------------------
void EmitCausalityPass::emitRecorderCall(OpBuilder &b, Location loc,
                                          const RecorderInfo &info,
                                          const DenseMap<Value, Value> &argMap) {
  // Runtime-invocation memoization: guard the call so the recorder body runs at
  // most ONCE per event. A reconvergent sub-cone is shared by many callers;
  // without this, the body (and its whole sub-tree) re-executes once per PATH
  // through the DAG — O(paths), catastrophic on wide cores (BOOM medium/large).
  // The guard collapses runtime cost to O(nodes). Output is byte-identical: the
  // predecessor contribution of a value is path-independent (see header), so
  // running once vs N-times-deduped yields the same trace.
  func::FuncOp recFn = info.funcOp; // copy the lightweight handle (info is const)
  Operation *fop = recFn.getOperation();
  auto it = recIdMap.find(fop);
  int64_t recId;
  if (it == recIdMap.end()) {
    recId = recIdCounter++;
    recIdMap[fop] = recId;
  } else {
    recId = it->second;
  }
  Value once =
      b.create<func::CallOp>(loc, recorderOnceDecl, ValueRange{cI64(b, loc, recId)})
          .getResult(0);
  Value first =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, once, cI32(b, loc, 0));
  auto ifOp = b.create<scf::IfOp>(loc, first, /*withElseRegion=*/false);
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(ifOp.thenBlock()->getTerminator());

  if (info.liveIns.empty()) {
    b.create<func::CallOp>(loc, info.funcOp, ValueRange{});
    return;
  }
  SmallVector<Value> callArgs;
  callArgs.reserve(info.liveIns.size());
  for (auto modelVal : info.liveIns) {
    auto ait = argMap.find(modelVal);
    assert(ait != argMap.end() &&
           "recorder live-in not found in caller argMap");
    callArgs.push_back(ait->second);
  }
  b.create<func::CallOp>(loc, info.funcOp, callArgs);
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

void EmitCausalityPass::emitAddCombPred(OpBuilder &b, Location loc,
                                         int64_t siteId, int64_t operand,
                                         Value isIdentityI8) {
  b.create<func::CallOp>(
      loc, addCombPredDecl,
      ValueRange{cI64(b, loc, siteId), cI64(b, loc, operand), isIdentityI8});
}

void EmitCausalityPass::emitAddMuxDecision(OpBuilder &b, Location loc,
                                            int64_t siteId, Value selValueI8,
                                            Value inputsDifferI8) {
  b.create<func::CallOp>(
      loc, addMuxDecisionDecl,
      ValueRange{cI64(b, loc, siteId), selValueI8, inputsDifferI8});
}

void EmitCausalityPass::emitCommit(OpBuilder &b, Location loc) {
  b.create<func::CallOp>(loc, commitDecl, ValueRange{});
}

// NEXT_STEPS #12: begin/add_pred variants taking a RUNTIME-computed i64 sink/pred
// id (cell = base + addr), vs the constant-id register variants above.
void EmitCausalityPass::emitBeginDynamic(OpBuilder &b, Location loc,
                                         Value sinkI64, int64_t wsid,
                                         Value newVal, int32_t numBits) {
  b.create<func::CallOp>(loc, beginDecl,
                         ValueRange{sinkI64, cI64(b, loc, wsid),
                                    toI64(b, loc, newVal), cI32(b, loc, numBits)});
}

void EmitCausalityPass::emitAddPredDynamic(OpBuilder &b, Location loc,
                                           Value predI64, int8_t role,
                                           int8_t delta) {
  b.create<func::CallOp>(
      loc, addPredDecl,
      ValueRange{predI64, cI8(b, loc, role), cI8(b, loc, delta)});
}

Value EmitCausalityPass::cellId(OpBuilder &b, Location loc, int64_t base,
                                Value addr) {
  return b.create<arith::AddIOp>(loc, cI64(b, loc, base), toI64(b, loc, addr));
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
