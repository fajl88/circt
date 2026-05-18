//===- InjectFault.cpp - Inject a single bit-flip fault at a write site ---===//
//
// Runs at the END of populateArcStateLoweringPipeline, immediately after
// EmitCausality.  At that point arc.state_write ops are still present and
// addressable by the write-site IDs assigned during EmitCausality::enumWriteSites().
//
// The pass enumerates arc.state_write ops in the same deterministic walk order
// as EmitCausality, locates the op at position faultWriteSiteId, then inserts:
//
//   %mask    = arith.constant (1 << fault_bit) : iN
//   %flipped = comb.xor %original_val, %mask   : iN
//
// before the write, replacing the written value with %flipped.
//
// Fails loudly (signalPassFailure + emitError) if:
//   - no write site exists at the requested position
//   - the written value is not an integer type
//   - fault_bit >= signal width
//
// Registration: manual PassWrapper (same strategy as EmitCausality) to avoid
// modifying ArcPasses.td and triggering a full rebuild.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/InjectFaultPass.h"

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

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
// runOnOperation
// ---------------------------------------------------------------------------

void InjectFaultPass::runOnOperation() {
  if (opts.faultWriteSiteId <= 0)
    return;

  ModuleOp module = getOperation();

  // Find the arc.state_write op at position faultWriteSiteId in module walk
  // order — same numbering EmitCausality assigns in enumWriteSites().
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

  // 3. Insert the bit-flip before the write.
  OpBuilder b(writeOp);
  Location loc = writeOp.getLoc();

  // mask = 1 << fault_bit  (same integer width as the written value)
  APInt maskVal(width, 1);
  maskVal <<= opts.faultBit;
  Value mask = arith::ConstantOp::create(b, loc, b.getIntegerAttr(itype, maskVal));

  // flipped = comb.xor writtenVal, mask
  Value flipped = comb::XorOp::create(b, loc, writtenVal, mask,
                                      /*twoState=*/true);

  // Replace the written value.
  writeOp.getValueMutable().assign(flipped);
}
