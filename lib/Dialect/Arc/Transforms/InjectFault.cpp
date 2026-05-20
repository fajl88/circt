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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"

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
// Helpers: enumerate comb.and / comb.or ops in the same deterministic walk
// order used by EmitCausality::enumCombSites().  Must stay in sync.
// ---------------------------------------------------------------------------

static Operation *findCombOpById(ModuleOp module, int64_t targetId) {
  int64_t counter = 1;
  Operation *found = nullptr;
  module.walk([&](Operation *op) {
    if (found)
      return;
    if (isa<comb::AndOp, comb::OrOp>(op)) {
      if (counter == targetId)
        found = op;
      counter++;
    }
  });
  return found;
}

// ---------------------------------------------------------------------------
// runOnOperation
// ---------------------------------------------------------------------------

void InjectFaultPass::runOnOperation() {
  const bool doBitFlip = opts.faultWriteSiteId > 0;
  const bool doGuardRemoval = opts.faultCombSiteId > 0;

  if (!doBitFlip && !doGuardRemoval)
    return;

  ModuleOp module = getOperation();

  if (doBitFlip && doGuardRemoval) {
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
  // Guard-removal: replace one operand of a comb.and / comb.or with the
  // gate's identity constant, permanently removing that operand's influence.
  //   AND identity: all-ones  (AND(all-ones, b...) == AND(b...))
  //   OR  identity: zero      (OR(zero,      b...) == OR(b...))
  // ------------------------------------------------------------------------
  Operation *targetOp = findCombOpById(module, opts.faultCombSiteId);
  if (!targetOp) {
    module.emitError("InjectFault: no comb site found with comb_site_id=")
        << opts.faultCombSiteId
        << " (max valid ID = number of comb.and/comb.or ops in the module)";
    signalPassFailure();
    return;
  }

  unsigned numOperands = targetOp->getNumOperands();
  if ((unsigned)opts.faultCombOperand >= numOperands) {
    module.emitError("InjectFault: comb_operand=")
        << opts.faultCombOperand << " is out of range for op with "
        << numOperands << " operands";
    signalPassFailure();
    return;
  }

  Value targetOperand = targetOp->getOperand(opts.faultCombOperand);
  auto itype = dyn_cast<IntegerType>(targetOperand.getType());
  if (!itype) {
    module.emitError("InjectFault: comb_site_id=")
        << opts.faultCombSiteId << " operand " << opts.faultCombOperand
        << " is not an IntegerType — cannot inject guard_removal";
    signalPassFailure();
    return;
  }

  const bool isAnd = isa<comb::AndOp>(targetOp);
  unsigned width = itype.getWidth();
  APInt identVal = isAnd ? APInt::getAllOnes(width) : APInt(width, 0);

  OpBuilder b(targetOp);
  Location loc = targetOp->getLoc();
  Value identConst = arith::ConstantOp::create(b, loc,
                                                b.getIntegerAttr(itype, identVal));
  targetOp->setOperand(opts.faultCombOperand, identConst);
}
