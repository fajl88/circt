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
  // NEXT_STEPS #12: comma-separated memory NAME tokens to instrument as
  // first-class, sliceable cells (cell_sink_id = mem_base + address). Empty =
  // OFF, in which case __signal_index.json + traces are byte-identical to the
  // register-only behaviour. A token matches a memory if it is a substring of
  // the memory's name attribute.
  std::string memoryNames;
};

/// Create an EmitCausalityPass instance.  If causalityDir is empty the pass
/// is a no-op (consistent with the conditional in arcilator.cpp).
std::unique_ptr<mlir::Pass>
createEmitCausalityPass(EmitCausalityPassOptions opts = {});

} // namespace arc
} // namespace circt
