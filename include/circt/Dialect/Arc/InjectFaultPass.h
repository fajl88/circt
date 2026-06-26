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
  int faultCombSiteId = 0;  // comb site ID for guard-removal injection (0 = disabled)
  int faultCombOperand = 0; // operand index to replace with identity (guard-removal only)
  // Runtime-switchable faults (NEXT_STEPS #6, coord/contracts/switchable_fault.md):
  // instead of baking ONE fault, make EVERY instrumentable site switchable at
  // runtime via a per-site i8 enable state. Classes: guard-removal (`__fault_en_<site>_<operand>`,
  // class byte 1) and forced mux-select (`__fault_en_mux_<site>`, classes 3/4) are
  // always instrumented. Mutually exclusive with the baked faultCombSiteId mode.
  bool faultSwitchable = false;
  // Eval-B #3 ablation: ALSO instrument every arc.state_write with a switchable naive
  // bit-flip (`__fault_en_bf_<site>`, the i8 value encodes the bit: 0=off, v=>bit v-1).
  // Built into a SEPARATE binary so the default switchable binary stays byte-identical
  // and cost-unchanged. Requires faultSwitchable. The naive strawman primitive
  // (DESIGN.md) — a loud opt-in, never on by default.
  bool faultSwitchableBitflip = false;
};

/// Create an InjectFaultPass instance.
/// With faultCombSiteId > 0 the pass bakes one guard-removal fault; with
/// faultSwitchable it makes every site runtime-switchable instead.
/// If neither is set the pass runs in "marked" mode (see below) and is a no-op
/// unless the module carries `trace.fault_target` attributes.
std::unique_ptr<mlir::Pass>
createInjectFaultPass(InjectFaultPassOptions opts = {});

/// Register InjectFault as a circt-opt pass (`--arc-inject-fault`).  Declared
/// here and called explicitly from circt-opt.cpp so the static registration is
/// not dropped by the linker (the Arc transforms lib is otherwise unreferenced
/// by circt-opt).  With no CLI options the pass runs in "marked" mode: it
/// removes the guard on every comb.and/comb.or carrying an integer
/// `trace.fault_target = <operand-index>` attribute.  This avoids needing
/// tablegen Pass::Options on a manually-declared pass.
void registerInjectFaultPass();

/// Register MaterializeCombWires as a circt-opt pass
/// (`--arc-materialize-comb-wires`).  Wraps every `trace.comb_site_id`-tagged
/// comb.and/comb.or result in a named, symbol-bearing `hw.wire`
/// (`__comb_site_<N>`) so the gate survives ExportVerilog as a standalone
/// Verilog net for the Encarsia miter cut wire, regardless of gate arity.
/// Run on the firtool/Verilog path only (the arcilator sim path is untouched).
void registerMaterializeCombWiresPass();

} // namespace arc
} // namespace circt
