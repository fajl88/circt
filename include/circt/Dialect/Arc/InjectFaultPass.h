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
  // Runtime-switchable faults (NEXT_STEPS #6, coord/contracts/switchable_fault.md):
  // instead of baking ONE fault, make EVERY trace.comb_site_id-tagged gate
  // operand switchable at runtime. Each (site, operand) gets an i8 model state
  // `__fault_en_<site>_<operand>` (a fault-CLASS byte: 0 = off, 1 = guard
  // removal; further classes reserved for NEXT_STEPS #6b) and the operand is
  // rewritten to `mux(class == 1, identity, operand)`. Mutually exclusive with
  // the baked modes above.
  bool faultSwitchable = false;
};

/// Create an InjectFaultPass instance.
/// Exactly one of faultWriteSiteId or faultCombSiteId must be nonzero.
/// If both are zero the pass runs in "marked" mode (see below) and is a no-op
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
