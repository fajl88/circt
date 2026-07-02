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

// Causality instrumentation mode. MANDATORY at build time (no default) so a
// forgotten flag can never silently pick a performance profile — the wrong
// choice here is a ~100x, hard-to-diagnose slowdown, so we fail loudly instead.
//   Injection: full predecessor-cone recording. The trace is backward-sliceable
//     (every arc.state_write records its combinational + sequential predecessors).
//     Required for the injection-pool programs we trace and slice.
//   Detection: value-only records (causality_begin/commit per write, but NO
//     predecessor cones). The fuzzer detection path compares observable VALUES
//     against Spike and never reads the cones, so cone recording there is pure
//     ~100x waste. __signal_index.json + the per-write VALUES are identical to
//     Injection; only the (unused-for-detection) predecessor lists are omitted.
enum class CausalityMode { Unset, Injection, Detection };

struct EmitCausalityPassOptions {
  std::string causalityDir; // output dir for __signal_index.json
  std::string sinkNames;    // comma-separated observable signal names
  // NEXT_STEPS #12: comma-separated memory NAME tokens to instrument as
  // first-class, sliceable cells (cell_sink_id = mem_base + address). Empty =
  // OFF, in which case __signal_index.json + traces are byte-identical to the
  // register-only behaviour. A token matches a memory if it is a substring of
  // the memory's name attribute.
  std::string memoryNames;
  // Predecessor-cone recording on (Injection) or off (Detection). Unset is an
  // error when causalityDir is set (see arcilator.cpp --causality-mode).
  CausalityMode mode = CausalityMode::Unset;
};

/// Create an EmitCausalityPass instance.  If causalityDir is empty the pass
/// is a no-op (consistent with the conditional in arcilator.cpp).
std::unique_ptr<mlir::Pass>
createEmitCausalityPass(EmitCausalityPassOptions opts = {});

} // namespace arc
} // namespace circt
