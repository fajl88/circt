// InjectFaultPass.h — public API for the InjectFault MLIR pass.
//
// Declared separately from ArcPasses.td (same strategy as EmitCausalityPass.h)
// to avoid the tablegen cascade rebuild.

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace arc {

struct InjectFaultPassOptions {
  int faultWriteSiteId = 0; // write site ID for bit-flip injection (0 = disabled)
  int faultBit = 0;         // bit position to flip (bit-flip only)
  int faultCombSiteId = 0;  // comb site ID for guard-removal injection (0 = disabled)
  int faultCombOperand = 0; // operand index to replace with identity (guard-removal only)
};

/// Create an InjectFaultPass instance.
/// Exactly one of faultWriteSiteId or faultCombSiteId must be nonzero.
/// If both are zero the pass is a no-op.
std::unique_ptr<mlir::Pass>
createInjectFaultPass(InjectFaultPassOptions opts = {});

} // namespace arc
} // namespace circt
