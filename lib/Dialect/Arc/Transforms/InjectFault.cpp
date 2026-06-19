//===- InjectFault.cpp - Inject a fault at a write or comb site -----------===//
//
// Runs at the END of populateArcStateLoweringPipeline, immediately after
// EmitCausality.  Supports two mutually exclusive fault modes:
//
// 1. Bit-flip (faultWriteSiteId > 0):
//    Locates the arc.state_write op at position faultWriteSiteId (same walk
//    order as EmitCausality::enumWriteSites()), then inserts:
//      %mask    = arith.constant (1 << fault_bit) : iN
//      %flipped = comb.xor %original_val, %mask   : iN
//    before the write, replacing the written value with %flipped.
//
// 2. Guard-removal (faultCombSiteId > 0):
//    Locates the comb.and / comb.or op at position faultCombSiteId (same walk
//    order as EmitCausality::enumCombSites()), then replaces operand
//    faultCombOperand with the gate's identity constant:
//      AND: all-ones  (identity for AND)
//      OR:  zero      (identity for OR)
//    This permanently removes the operand from the gate's influence, modelling
//    a forgotten-guard structural bug.
//
// Fails loudly (signalPassFailure + emitError) if both IDs are nonzero,
// if the requested site is not found, or if the operand is out of range.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/InjectFaultPass.h"

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <utility>

using namespace circt;
using namespace arc;
using namespace mlir;

namespace {

class InjectFaultPass
    : public PassWrapper<InjectFaultPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InjectFaultPass)

  InjectFaultPass() = default;
  InjectFaultPass(const InjectFaultPass &) = default;
  explicit InjectFaultPass(circt::arc::InjectFaultPassOptions opts)
      : opts(std::move(opts)) {}

  StringRef getArgument() const final { return "arc-inject-fault"; }
  StringRef getDescription() const final {
    return "Inject a bit-flip fault at a specific arc.state_write site";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, comb::CombDialect>();
  }

  void runOnOperation() override;

private:
  circt::arc::InjectFaultPassOptions opts;
};

} // namespace

std::unique_ptr<Pass>
circt::arc::createInjectFaultPass(InjectFaultPassOptions opts) {
  return std::make_unique<InjectFaultPass>(std::move(opts));
}

// ---------------------------------------------------------------------------
// Helpers: enumerate arc.state_write ops in the same deterministic walk order
// used by EmitCausality::enumWriteSites().  Must stay in sync.
// ---------------------------------------------------------------------------

static StateWriteOp findWriteOpById(ModuleOp module, int64_t targetId) {
  int64_t counter = 1;
  StateWriteOp found;
  module.walk([&](StateWriteOp op) {
    if (found)
      return;
    if (counter == targetId)
      found = op;
    counter++;
  });
  return found;
}

// ---------------------------------------------------------------------------
// Helpers: locate comb.and / comb.or guard-removal targets.
//
// Guard-removal is source-anchored: each comb.and/comb.or carries a stable
// `trace.comb_site_id` attribute (assigned at the top.mlir level, before arc
// lowering / inlining, by EmitCausality's tagging step).  A single source gate
// may be inlined into many copies, all sharing the same id; faulting "comb site
// N" means faulting EVERY copy carrying that id, modelling a forgotten guard on
// all uses of one RTL gate.
// ---------------------------------------------------------------------------

// Collect every comb.and/comb.or whose `trace.comb_site_id` equals targetId.
static void findCombOpsBySiteId(ModuleOp module, int64_t targetId,
                                SmallVectorImpl<Operation *> &out) {
  module.walk([&](Operation *op) {
    if (!isa<comb::AndOp, comb::OrOp>(op))
      return;
    if (auto a = op->getAttrOfType<IntegerAttr>("trace.comb_site_id"))
      if (a.getInt() == targetId)
        out.push_back(op);
  });
}

// Replace operand `operand` of an AND/OR gate with the gate's identity constant
// (all-ones for AND, zero for OR), permanently removing it.  Returns failure
// (after emitting an error) if the operand is out of range or non-integer.
static LogicalResult faultCombOp(Operation *targetOp, int operand,
                                 int64_t idForMsg) {
  unsigned numOperands = targetOp->getNumOperands();
  if ((unsigned)operand >= numOperands) {
    targetOp->emitError("InjectFault: comb_operand=")
        << operand << " is out of range for op with " << numOperands
        << " operands (comb_site_id=" << idForMsg << ")";
    return failure();
  }
  auto itype = dyn_cast<IntegerType>(targetOp->getOperand(operand).getType());
  if (!itype) {
    targetOp->emitError("InjectFault: comb_site_id=")
        << idForMsg << " operand " << operand
        << " is not an IntegerType — cannot inject guard_removal";
    return failure();
  }
  const bool isAnd = isa<comb::AndOp>(targetOp);
  unsigned width = itype.getWidth();
  APInt identVal = isAnd ? APInt::getAllOnes(width) : APInt(width, 0);
  OpBuilder b(targetOp);
  // hw.constant (not arith.constant): the faulted module is lowered by BOTH
  // arcilator (sim) and firtool/ExportVerilog (formal miter). firtool's --verilog
  // pipeline does not load the arith dialect, so an arith.constant left here makes
  // ExportVerilog fail ("Dialect `arith' not found"). hw.constant is native to the
  // HW/Comb level and lowers cleanly on both paths.
  Value identConst =
      hw::ConstantOp::create(b, targetOp->getLoc(), identVal);
  targetOp->setOperand(operand, identConst);
  return success();
}

// Collect the source mux ids encoded in a synthetic location (#5): a FileLineColLoc
// with filename "mux_site" carries the source id in its line field (stamped by
// tools/tag_comb_sites.py tag-mux). Mux attributes do NOT survive arc lowering, but the
// LOCATION does, so this recovers a cone mux's source id post-lowering. Recurses
// FusedLoc / NameLoc; returns the SET of distinct ids (a clean source mux → exactly 1).
// Mirrors EmitCausality::collectMuxSiteIds (the recorder side); kept in sync.
static void collectMuxSiteIds(Location loc,
                              llvm::SmallDenseSet<int64_t, 2> &ids) {
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

// Baked forced-mux-select (broken conditional, NEXT_STEPS #5): replace a marked
// comb.mux's SELECT (operand 0) with a constant i1, permanently forcing the mux to
// take ONE arm — the canonical "broken conditional" (a forgotten/stuck guard).
//
// Unlike guard-removal, this MUST be injected on top_tagged.mlir BEFORE arc
// lowering, where the source mux still carries its unique trace.mux_site_id: mux
// attributes do NOT survive arc lowering (canonicalization/inlining rebuild the
// register cones untagged — exactly why the recorder anchors on the synthetic
// location instead). Forcing the select on the SOURCE mux and re-lowering lets the
// optimizer propagate the corruption through inlining/merging faithfully, so this is
// correct for ANY source mux, including ones a later pass would merge. comb.mux is
// `sel ? trueVal : falseVal`, so forcedBit=1 forces the TRUE arm, 0 the FALSE arm.
static LogicalResult faultMuxOp(Operation *targetOp, int forcedBit,
                                int64_t idForMsg) {
  auto mux = dyn_cast<comb::MuxOp>(targetOp);
  if (!mux) {
    targetOp->emitError("InjectFault: mux_site_id=")
        << idForMsg << " marked op is not a comb.mux";
    return failure();
  }
  if (forcedBit != 0 && forcedBit != 1) {
    targetOp->emitError("InjectFault: mux_site_id=")
        << idForMsg << " force value " << forcedBit
        << " is not 0 or 1 (the select is one bit)";
    return failure();
  }
  OpBuilder b(targetOp);
  // hw.constant (not arith.constant): the faulted module is lowered by BOTH arcilator
  // (sim) and firtool/ExportVerilog (miter); arith isn't loaded on the verilog path.
  Value sel = hw::ConstantOp::create(b, targetOp->getLoc(), APInt(1, forcedBit));
  targetOp->setOperand(0, sel); // operand 0 of comb.mux is the select
  return success();
}

// ---------------------------------------------------------------------------
// Switchable mode (NEXT_STEPS #6, coord/contracts/switchable_fault.md).
//
// Two fault classes share ONE binary, selected at runtime by an i8 class byte:
//
//   GUARD REMOVAL (class 1) — every trace.comb_site_id-tagged comb.and/comb.or
//   operand becomes runtime-switchable:
//     %en   = arc.state_read %__fault_en_<site>_<k> : i8
//     %on   = comb.icmp eq %en, 1
//     %op'  = comb.mux %on, <identity>, %operand
//
//   FORCED MUX-SELECT (classes 3/4, the broken conditional, NEXT_STEPS #5) —
//   every clean source comb.mux has its select runtime-gated:
//     %en   = arc.state_read %__fault_en_mux_<N> : i8
//     %sel' = (en==3) ? 0 : (en==4) ? 1 : %origSel   // 3 = force false arm, 4 = force true arm
//   The source mux is identified by its synthetic loc("mux_site":N) — mux
//   attributes do NOT survive arc lowering, but the location does, so all inlined
//   clones of source mux N carry id N and share __fault_en_mux_<N>: enabling N
//   faults every descendant, the runtime analog of the baked faultMuxOp.
//
// The i8 enable states are allocated in the enclosing arc.model body (this pass
// runs BEFORE state allocation, so AllocateState assigns offsets and ModelInfo
// exports them to model_state.json by name, where the driver resolves and pokes
// them). EmitCausality already ran, so with all enables 0 the inserted muxes pass
// the originals through and the model is bit-identical to the reference. Class
// bytes are compared exactly (==1, ==3, ==4) so the classes never alias.
// ---------------------------------------------------------------------------

static LogicalResult runSwitchable(ModuleOp module) {
  // Switchable sites must live inside an arc.model body — that is where the
  // storage block argument for the enable states lives. A site outside any model
  // could not be switched and would silently diverge from the baked mode, a hard
  // error.
  DenseMap<ModelOp, SmallVector<Operation *>> gatesByModel;
  DenseMap<ModelOp, SmallVector<Operation *>> muxesByModel; // forced mux-select (#5)
  int64_t taggedTotal = 0, taggedMuxTotal = 0;
  WalkResult walkRes = module.walk([&](Operation *op) -> WalkResult {
    // comb.and / comb.or guard-removal sites (by the surviving trace.comb_site_id attr).
    if (isa<comb::AndOp, comb::OrOp>(op) &&
        op->hasAttrOfType<IntegerAttr>("trace.comb_site_id")) {
      ++taggedTotal;
      auto model = op->getParentOfType<ModelOp>();
      if (!model) {
        op->emitError("InjectFault(switchable): tagged comb gate (comb_site_id=")
            << op->getAttrOfType<IntegerAttr>("trace.comb_site_id").getInt()
            << ") is not inside an arc.model — cannot allocate its enable state";
        return WalkResult::interrupt();
      }
      gatesByModel[model].push_back(op);
      return WalkResult::advance();
    }
    // forced mux-select sites: a comb.mux whose synthetic location names exactly ONE
    // source mux id (the clean 99.1%; fused/synthesized muxes can't be cleanly
    // attributed, so they are skipped — matching the recorder's exclusion).
    if (isa<comb::MuxOp>(op)) {
      llvm::SmallDenseSet<int64_t, 2> ids;
      collectMuxSiteIds(op->getLoc(), ids);
      if (ids.size() == 1) {
        ++taggedMuxTotal;
        auto model = op->getParentOfType<ModelOp>();
        if (!model) {
          op->emitError("InjectFault(switchable): source mux (mux_site_id=")
              << *ids.begin()
              << ") is not inside an arc.model — cannot allocate its enable state";
          return WalkResult::interrupt();
        }
        muxesByModel[model].push_back(op);
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  if (walkRes.wasInterrupted())
    return failure();

  if (taggedTotal == 0 && taggedMuxTotal == 0) {
    module.emitError(
        "InjectFault(switchable): no trace.comb_site_id-tagged gates and no "
        "mux_site-located muxes found (was the input run through tag + tag-mux?)");
    return failure();
  }

  int64_t numEnableStates = 0, numMuxedOperands = 0;
  for (auto &[model, gates] : gatesByModel) {
    Block &body = model.getBodyBlock();
    if (body.getNumArguments() < 1) {
      model.emitError("InjectFault(switchable): arc.model body has no storage "
                      "block argument");
      return failure();
    }
    Value storage = body.getArgument(0);

    OpBuilder allocBuilder(model.getContext());
    allocBuilder.setInsertionPointToStart(&body);
    auto i8Ty = allocBuilder.getIntegerType(8);

    // One enable state per (site, operand); clones share it. Arity must agree
    // across clones of one site (clones are copies of one source gate).
    DenseMap<std::pair<int64_t, int64_t>, Value> enableState;
    DenseMap<int64_t, unsigned> arityBySite;

    for (Operation *gate : gates) {
      int64_t siteId =
          gate->getAttrOfType<IntegerAttr>("trace.comb_site_id").getInt();
      unsigned numOperands = gate->getNumOperands();
      auto [it, first] = arityBySite.try_emplace(siteId, numOperands);
      if (!first && it->second != numOperands) {
        gate->emitError("InjectFault(switchable): clones of comb_site_id=")
            << siteId << " disagree on operand count (" << it->second
            << " vs " << numOperands << ") — refusing to switch inconsistently";
        return failure();
      }

      const bool isAnd = isa<comb::AndOp>(gate);
      OpBuilder b(gate);
      Location loc = gate->getLoc();
      for (unsigned k = 0; k < numOperands; ++k) {
        auto key = std::make_pair(siteId, (int64_t)k);
        Value &state = enableState[key];
        if (!state) {
          auto alloc = AllocStateOp::create(allocBuilder, model.getLoc(),
                                            StateType::get(i8Ty), storage);
          alloc->setAttr("name",
                         allocBuilder.getStringAttr(
                             "__fault_en_" + std::to_string(siteId) + "_" +
                             std::to_string(k)));
          state = alloc;
          ++numEnableStates;
        }

        auto itype = dyn_cast<IntegerType>(gate->getOperand(k).getType());
        if (!itype) {
          gate->emitError("InjectFault(switchable): comb_site_id=")
              << siteId << " operand " << k << " is not an IntegerType";
          return failure();
        }
        unsigned width = itype.getWidth();
        APInt identVal = isAnd ? APInt::getAllOnes(width) : APInt(width, 0);

        Value en = StateReadOp::create(b, loc, state);
        Value clsGuardRemoval = hw::ConstantOp::create(b, loc, APInt(8, 1));
        Value on = comb::ICmpOp::create(b, loc, comb::ICmpPredicate::eq, en,
                                        clsGuardRemoval);
        Value ident = hw::ConstantOp::create(b, loc, identVal);
        Value switched =
            comb::MuxOp::create(b, loc, on, ident, gate->getOperand(k));
        gate->setOperand(k, switched);
        ++numMuxedOperands;
      }
    }
  }

  // Forced mux-select (#5): gate each clean source mux's select on __fault_en_mux_<N>.
  // en==3 forces the select to 0 (false arm), en==4 to 1 (true arm); en==0 (and the
  // guard-removal class 1) leave the original select untouched, bit-identical to ref.
  int64_t numMuxEnableStates = 0, numGatedMuxes = 0;
  for (auto &[model, muxes] : muxesByModel) {
    Block &body = model.getBodyBlock();
    if (body.getNumArguments() < 1) {
      model.emitError("InjectFault(switchable): arc.model body has no storage "
                      "block argument");
      return failure();
    }
    Value storage = body.getArgument(0);
    OpBuilder allocBuilder(model.getContext());
    allocBuilder.setInsertionPointToStart(&body);
    auto i8Ty = allocBuilder.getIntegerType(8);

    DenseMap<int64_t, Value> muxEnableState; // per source mux id; clones share
    for (Operation *op : muxes) {
      llvm::SmallDenseSet<int64_t, 2> ids;
      collectMuxSiteIds(op->getLoc(), ids);
      int64_t siteId = *ids.begin();
      Value &state = muxEnableState[siteId];
      if (!state) {
        auto alloc = AllocStateOp::create(allocBuilder, model.getLoc(),
                                          StateType::get(i8Ty), storage);
        alloc->setAttr("name", allocBuilder.getStringAttr(
                                   "__fault_en_mux_" + std::to_string(siteId)));
        state = alloc;
        ++numMuxEnableStates;
      }

      auto mux = cast<comb::MuxOp>(op);
      OpBuilder b(op);
      Location loc = op->getLoc();
      Value origSel = mux.getCond();
      Value en = StateReadOp::create(b, loc, state);
      Value c3 = hw::ConstantOp::create(b, loc, APInt(8, 3));
      Value c4 = hw::ConstantOp::create(b, loc, APInt(8, 4));
      Value is3 = comb::ICmpOp::create(b, loc, comb::ICmpPredicate::eq, en, c3);
      Value is4 = comb::ICmpOp::create(b, loc, comb::ICmpPredicate::eq, en, c4);
      Value sel0 = hw::ConstantOp::create(b, loc, APInt(1, 0));
      Value sel1 = hw::ConstantOp::create(b, loc, APInt(1, 1));
      // en==4 -> 1 (true arm) else original; then en==3 -> 0 (false arm).
      Value selT = comb::MuxOp::create(b, loc, is4, sel1, origSel);
      Value selF = comb::MuxOp::create(b, loc, is3, sel0, selT);
      mux.setOperand(0, selF);
      ++numGatedMuxes;
    }
  }

  llvm::errs() << "[inject-fault] switchable: " << taggedTotal
               << " tagged gate clones, " << numEnableStates
               << " __fault_en_<site>_<operand> states, " << numMuxedOperands
               << " operands muxed; " << taggedMuxTotal << " source-mux clones, "
               << numMuxEnableStates << " __fault_en_mux_<site> states, "
               << numGatedMuxes << " mux selects gated\n";
  return success();
}

// ---------------------------------------------------------------------------
// runOnOperation
// ---------------------------------------------------------------------------

void InjectFaultPass::runOnOperation() {
  const bool doBitFlip = opts.faultWriteSiteId > 0;

  ModuleOp module = getOperation();

  if (doBitFlip && opts.faultCombSiteId > 0) {
    module.emitError("InjectFault: faultWriteSiteId and faultCombSiteId are "
                     "mutually exclusive — set exactly one");
    signalPassFailure();
    return;
  }

  if (opts.faultSwitchable) {
    if (doBitFlip || opts.faultCombSiteId > 0) {
      module.emitError("InjectFault: faultSwitchable is mutually exclusive "
                       "with the baked faultWriteSiteId/faultCombSiteId modes");
      signalPassFailure();
      return;
    }
    if (failed(runSwitchable(module)))
      signalPassFailure();
    return;
  }

  if (doBitFlip) {
    // ------------------------------------------------------------------
    // Bit-flip: XOR one bit of the written value before the write.
    // ------------------------------------------------------------------
    StateWriteOp writeOp = findWriteOpById(module, opts.faultWriteSiteId);
    if (!writeOp) {
      module.emitError("InjectFault: no write site found with write_site_id=")
          << opts.faultWriteSiteId
          << " (max valid ID = number of arc.state_write ops in the module)";
      signalPassFailure();
      return;
    }

    Value writtenVal = writeOp.getValue();
    auto itype = dyn_cast<IntegerType>(writtenVal.getType());
    if (!itype) {
      module.emitError("InjectFault: write_site_id=")
          << opts.faultWriteSiteId
          << " written value is not an IntegerType — cannot inject bit_flip";
      signalPassFailure();
      return;
    }

    unsigned width = itype.getWidth();
    if (opts.faultBit < 0 || (unsigned)opts.faultBit >= width) {
      module.emitError("InjectFault: fault_bit=")
          << opts.faultBit << " is out of range for " << width << "-bit signal";
      signalPassFailure();
      return;
    }

    OpBuilder b(writeOp);
    Location loc = writeOp.getLoc();
    APInt maskVal(width, 1);
    maskVal <<= opts.faultBit;
    Value mask = arith::ConstantOp::create(b, loc,
                                            b.getIntegerAttr(itype, maskVal));
    Value flipped = comb::XorOp::create(b, loc, writtenVal, mask,
                                         /*twoState=*/true);
    writeOp.getValueMutable().assign(flipped);
    return;
  }

  // ------------------------------------------------------------------------
  // Guard-removal (source-anchored).  Replace one operand of a comb.and /
  // comb.or with the gate's identity constant (all-ones for AND, zero for OR),
  // permanently removing that operand's influence.  Target selection has two
  // modes:
  //   (a) opts.faultCombSiteId > 0 (e.g. arcilator --fault-comb-site-id): fault
  //       EVERY comb gate tagged trace.comb_site_id == faultCombSiteId, operand
  //       = faultCombOperand.  A single source gate inlined into K copies is
  //       faulted on all K (one wrong RTL line affects all its uses).
  //   (b) opts.faultCombSiteId == 0 (circt-opt --arc-inject-fault, no options):
  //       fault every comb gate carrying trace.fault_target = <operand-index>.
  //       Lets a faulted top.mlir be produced for ExportVerilog without needing
  //       tablegen Pass::Options on this manually-declared pass.
  // ------------------------------------------------------------------------
  SmallVector<std::pair<Operation *, int>> targets;
  if (opts.faultCombSiteId > 0) {
    SmallVector<Operation *> ops;
    findCombOpsBySiteId(module, opts.faultCombSiteId, ops);
    if (ops.empty()) {
      module.emitError("InjectFault: no comb gate tagged trace.comb_site_id=")
          << opts.faultCombSiteId
          << " (was the input run through the comb-site tagging step?)";
      signalPassFailure();
      return;
    }
    for (auto *op : ops)
      targets.push_back({op, opts.faultCombOperand});
  } else {
    SmallVector<Operation *> muxMarked; // forced-mux-select (NEXT_STEPS #5)
    module.walk([&](Operation *op) {
      if (isa<comb::AndOp, comb::OrOp>(op)) {
        if (auto a = op->getAttrOfType<IntegerAttr>("trace.fault_target"))
          targets.push_back({op, static_cast<int>(a.getInt())});
      } else if (isa<comb::MuxOp>(op) &&
                 op->hasAttrOfType<IntegerAttr>("trace.mux_force") &&
                 op->hasAttrOfType<IntegerAttr>("trace.mux_site_id")) {
        // Baked forced-mux-select: a marker (trace.mux_force = 0|1) on the source
        // mux at top_tagged.mlir, forced PRE-arc-lowering (where the source mux is
        // the one op carrying trace.mux_site_id). arcilator then lowers the already-
        // corrupted IR. This is the broken-conditional injection (NEXT_STEPS #5).
        muxMarked.push_back(op);
      }
    });
    // Marked mode with nothing marked is a deliberate pass-through no-op.
    if (targets.empty() && muxMarked.empty())
      return;
    for (Operation *op : muxMarked) {
      int64_t id = op->getAttrOfType<IntegerAttr>("trace.mux_site_id").getInt();
      int force = static_cast<int>(
          op->getAttrOfType<IntegerAttr>("trace.mux_force").getInt());
      if (failed(faultMuxOp(op, force, id))) {
        signalPassFailure();
        return;
      }
    }
  }

  for (auto &t : targets) {
    int64_t idForMsg = opts.faultCombSiteId;
    if (auto a = t.first->getAttrOfType<IntegerAttr>("trace.comb_site_id"))
      idForMsg = a.getInt();
    if (failed(faultCombOp(t.first, t.second, idForMsg))) {
      signalPassFailure();
      return;
    }
  }
}

void circt::arc::registerInjectFaultPass() {
  ::mlir::PassRegistration<InjectFaultPass>();
}

// ===========================================================================
// MaterializeCombWires — make every tagged comb gate a named Verilog cut wire.
//
// A guard-removal fault replaces one operand of a comb.and/comb.or with the
// gate's identity. For a 2-input gate that leaves a single operand, so the gate
// becomes a pure passthrough that firtool folds away — and the cut wire the
// Encarsia miter needs (`__comb_site_<N>`) vanishes in the faulted Verilog.
//
// This pass wraps each `trace.comb_site_id`-tagged gate's result in an
// `hw.wire` carrying BOTH a name ("__comb_site_<N>") and an inner symbol
// (@__comb_site_<N>). The symbol is load-bearing: a name-only hw.wire folds
// (its name degrades to a hint that can land on the wrong net), whereas a
// symbol-bearing wire is a hard barrier ExportVerilog must keep — so the named
// net survives in BOTH top_ref.sv and top_fault.sv regardless of gate arity.
//
// Runs ONLY on the firtool/ExportVerilog path (on top_named.mlir, before
// firtool); the arcilator-fed top_tagged.mlir is untouched, so the simulation
// path and signal index are unchanged. Supersedes the earlier sv.namehint +
// wireSpilling approach, which could not preserve folded (2-input) gates.
// ===========================================================================

namespace {
class MaterializeCombWiresPass
    : public PassWrapper<MaterializeCombWiresPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeCombWiresPass)

  StringRef getArgument() const final { return "arc-materialize-comb-wires"; }
  StringRef getDescription() const final {
    return "Wrap each trace.comb_site_id-tagged comb.and/comb.or result in a "
           "named, symbol-bearing hw.wire (__comb_site_<N>) so it survives as a "
           "standalone Verilog net for the formal miter, independent of arity";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hw::HWDialect, comb::CombDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void MaterializeCombWiresPass::runOnOperation() {
  ModuleOp module = getOperation();
  // Collect first; we insert ops + reroute uses below, so don't mutate mid-walk.
  SmallVector<Operation *> gates;
  module.walk([&](Operation *op) {
    if (isa<comb::AndOp, comb::OrOp>(op) &&
        op->hasAttrOfType<IntegerAttr>("trace.comb_site_id"))
      gates.push_back(op);
  });
  for (Operation *op : gates) {
    int64_t id = op->getAttrOfType<IntegerAttr>("trace.comb_site_id").getInt();
    Value res = op->getResult(0);
    OpBuilder b(op);
    b.setInsertionPointAfter(op);
    StringAttr nameAttr = b.getStringAttr("__comb_site_" + std::to_string(id));
    auto wire = hw::WireOp::create(b, op->getLoc(), res, nameAttr,
                                   hw::InnerSymAttr::get(nameAttr));
    // Reroute every other reader through the wire (the wire itself still reads
    // the gate result).
    res.replaceAllUsesExcept(wire.getResult(), wire.getOperation());
  }
  // Mux cut wires (NEXT_STEPS #5 forced-mux): wrap each tagged SOURCE comb.mux
  // result in a __mux_site_<N> symbol-bearing hw.wire, so the JG-miter cut leaf
  // survives in both ref and forced-mux Verilog independent of how the cone lowers.
  SmallVector<Operation *> muxes;
  module.walk([&](Operation *op) {
    if (isa<comb::MuxOp>(op) &&
        op->hasAttrOfType<IntegerAttr>("trace.mux_site_id"))
      muxes.push_back(op);
  });
  for (Operation *op : muxes) {
    int64_t id = op->getAttrOfType<IntegerAttr>("trace.mux_site_id").getInt();
    Value res = op->getResult(0);
    OpBuilder b(op);
    b.setInsertionPointAfter(op);
    StringAttr nameAttr = b.getStringAttr("__mux_site_" + std::to_string(id));
    auto wire = hw::WireOp::create(b, op->getLoc(), res, nameAttr,
                                   hw::InnerSymAttr::get(nameAttr));
    res.replaceAllUsesExcept(wire.getResult(), wire.getOperation());
  }
  // Strip ALL trace.* attrs once the wires carry the ids. This is load-bearing
  // for CORRECTNESS, not cosmetic: ExportVerilog's PrepareForEmission refuses
  // to binarize a variadic comb op carrying an attr from an unregistered
  // dialect ("trace" is not a dialect), and the release-built expression
  // printer then silently emits only the FIRST TWO operands of a >=3-operand
  // variadic op (its numOperands==2 assert is compiled out) — miscompiling the
  // exported Verilog. This pass is the last trace-aware step on the firtool
  // path (run AFTER mark/inject), so it sweeps every op.
  int64_t stripped = 0;
  module.walk([&](Operation *op) {
    for (StringRef name :
         {"trace.comb_site_id", "trace.fault_target",
          "trace.mux_site_id", "trace.mux_force"})
      if (op->removeAttr(name))
        ++stripped;
  });
  llvm::errs() << "[materialize-comb-wires] wrapped " << gates.size()
               << " tagged comb gates in named hw.wire (__comb_site_<N>), "
               << "stripped " << stripped << " trace.* attrs\n";
}

void circt::arc::registerMaterializeCombWiresPass() {
  ::mlir::PassRegistration<MaterializeCombWiresPass>();
}
