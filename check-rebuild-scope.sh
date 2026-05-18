#!/usr/bin/env bash
# check-rebuild-scope.sh — Preview how many ninja steps a file change would trigger.
#
# Usage:
#   third_party/circt/check-rebuild-scope.sh file1 [file2 ...]
#
# How it works:
#   1. Records the current mtime of each file.
#   2. Touches the files (makes them newer than build artifacts).
#   3. Runs `ninja -n bin/arcilator` (dry-run — no actual compilation).
#   4. Counts and categorises the planned steps.
#   5. Restores original mtimes (leaves the repo in exactly the same state).
#   6. Prints a summary and exits non-zero if the scope is dangerous (>5 steps).
#
# This script is safe to run at any time — it never modifies build outputs.
# Run it BEFORE every `ninja`/`cmake --build` invocation after changing CIRCT source.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 file1 [file2 ...]"
    exit 1
fi

BUILD=$(cd "$(dirname "$0")/build" && pwd)

if [ ! -f "$BUILD/build.ninja" ]; then
    echo "ERROR: build directory not found at $BUILD"
    exit 1
fi

# --------------------------------------------------------------------------
# 1. Record original mtimes
# --------------------------------------------------------------------------
declare -A ORIG_MTIME
for f in "$@"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: file not found: $f"
        exit 1
    fi
    ORIG_MTIME["$f"]=$(stat -c %Y "$f")
done

# --------------------------------------------------------------------------
# 2. Touch files to simulate a modification
# --------------------------------------------------------------------------
for f in "$@"; do
    touch "$f"
done

# --------------------------------------------------------------------------
# 3. Dry-run and capture the plan
# --------------------------------------------------------------------------
PLAN=$(ninja -C "$BUILD" -n bin/arcilator 2>&1 || true)

# --------------------------------------------------------------------------
# 4. Restore original mtimes — ALWAYS, even if ninja -n failed
# --------------------------------------------------------------------------
for f in "$@"; do
    touch -d "@${ORIG_MTIME[$f]}" "$f"
done

# --------------------------------------------------------------------------
# 5. Analyse and report
# --------------------------------------------------------------------------
TOTAL=$(echo "$PLAN" | grep -cE '^\[' || true)
TABLEGEN=$(echo "$PLAN" | grep -cE 'tablegen|\.inc' || true)
COMPILE=$(echo "$PLAN" | grep -cE 'Building CXX' || true)
LINK=$(echo "$PLAN" | grep -cE 'Linking' || true)

echo ""
echo "Rebuild scope for: $*"
echo "  Total ninja steps : $TOTAL"
echo "  Tablegen steps    : $TABLEGEN"
echo "  C++ compile steps : $COMPILE"
echo "  Link steps        : $LINK"
echo ""

if [ "$TABLEGEN" -gt 0 ]; then
    echo "DANGER: tablegen would run. This means a .td file or a header that"
    echo "        tablegen depends on has changed. All Arc transforms (~30 files)"
    echo "        will be recompiled (1–2 hour rebuild)."
    echo ""
    echo "        DO NOT proceed with the build. Revert the .td change and use"
    echo "        a different approach (e.g. PassWrapper instead of tablegen)."
    echo ""
    echo "        See third_party/circt/lib/Dialect/Arc/Transforms/EmitCausality.cpp"
    echo "        for an example of how to register a pass without touching .td files."
elif [ "$TOTAL" -gt 5 ]; then
    echo "WARNING: $TOTAL steps planned. This is more than expected for a single"
    echo "         file change (expected: 1–2 steps: compile + link, or link only)."
    echo ""
    echo "         Likely causes:"
    echo "           - A header included by many files was modified."
    echo "           - A previous ninja build was killed mid-run (corrupted mtimes)."
    echo ""
    echo "         If the build is corrupted, the only fix is a full rebuild."
    echo "         If this is unexpected, revert your changes and check again."
else
    echo "OK: $TOTAL step(s) planned. Safe to build."
    echo "    Run: ninja -C $BUILD bin/arcilator"
fi

if [ "$TOTAL" -gt 5 ] || [ "$TABLEGEN" -gt 0 ]; then
    exit 1
fi
