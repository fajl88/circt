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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
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
  Value identConst = arith::ConstantOp::create(
      b, targetOp->getLoc(), b.getIntegerAttr(itype, identVal));
  targetOp->setOperand(operand, identConst);
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
    module.walk([&](Operation *op) {
      if (!isa<comb::AndOp, comb::OrOp>(op))
        return;
      if (auto a = op->getAttrOfType<IntegerAttr>("trace.fault_target"))
        targets.push_back({op, static_cast<int>(a.getInt())});
    });
    // Marked mode with nothing marked is a deliberate pass-through no-op.
    if (targets.empty())
      return;
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
