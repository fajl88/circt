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
  int faultWriteSiteId = 0; // signal ID to inject into (0 = disabled)
  int faultBit = 0;         // bit position to flip
};

/// Create an InjectFaultPass instance.  If faultWriteSiteId <= 0 the pass
/// is a no-op.
std::unique_ptr<mlir::Pass>
createInjectFaultPass(InjectFaultPassOptions opts = {});

} // namespace arc
} // namespace circt
