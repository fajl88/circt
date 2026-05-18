// EmitCausalityPass.h — public API for the EmitCausality MLIR pass.
//
// Declared separately from ArcPasses.td so that only arcilator.cpp (and
// EmitCausality.cpp itself) need to include this header, avoiding the
// tablegen-cascading rebuild that modifying ArcPasses.td would cause.

#pragma once

#include <memory>
#include <string>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace arc {

struct EmitCausalityPassOptions {
  std::string causalityDir; // output dir for __signal_index.json
  std::string sinkNames;    // comma-separated observable signal names
};

/// Create an EmitCausalityPass instance.  If causalityDir is empty the pass
/// is a no-op (consistent with the conditional in arcilator.cpp).
std::unique_ptr<mlir::Pass>
createEmitCausalityPass(EmitCausalityPassOptions opts = {});

} // namespace arc
} // namespace circt
