#!/bin/bash
# One benchmark array element: work out which implementation this index runs, then run it.
#
#   run_arm.sh stats   <camera> <chunks_per_job>
#   run_arm.sh correct <setup> <arm> [<arm> ...]      # arm is "impl:semaphore"
#
# This exists as a file rather than inline in the bsub line because everything here
# depends on $LSB_JOBINDEX, which must expand in the JOB's shell. Inlined, the arithmetic
# sits inside the quoting of a `bsub ... "pixi run bash -c '...'"` sandwich, and one
# wrong layer expands it at submit time -- where the variable is empty and
# `$(( $PAIR * 64 + 1 ))` becomes a syntax error.

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$1"; shift
IDX="${LSB_JOBINDEX:-1}"

# Pin both implementations to the same thread counts. They drive the same tensorstore
# underneath, so leaving these at their defaults benchmarks tensorstore's defaults
# rather than the port.
CORES="${LSB_DJOB_NUMPROC:-8}"
export JULIA_NUM_THREADS="$CORES"
export OMP_NUM_THREADS="$CORES"
export OPENBLAS_NUM_THREADS="$CORES"
export MKL_NUM_THREADS="$CORES"

case "$STAGE" in
  stats)
    CAMERA="$1"; PER_JOB="$2"
    # Index parity picks the implementation; each PAIR of elements shares one chunk
    # range, so the two arms can be compared directly instead of across submissions.
    PAIR=$(( (IDX - 1) / 2 ))
    if [ $(( IDX % 2 )) -eq 0 ]; then IMPL=julia; else IMPL=python; fi
    START=$(( PAIR * PER_JOB + 1 ))
    STOP=$(( START + PER_JOB - 1 ))
    echo "index $IDX -> $IMPL stats camera $CAMERA chunks $START..$STOP on $(hostname)"
    exec python "$HERE/bench.py" "$IMPL" stats "$CAMERA" "$START" "$STOP"
    ;;
  correct)
    SETUP="$1"; shift
    ARMS=("$@")
    A="${ARMS[$(( (IDX - 1) % ${#ARMS[@]} ))]}"
    IMPL="${A%%:*}"
    export SPOTLIGHT_CORRECT_CONCURRENCY="${A##*:}"
    echo "index $IDX -> $IMPL correct setup $SETUP sem=$SPOTLIGHT_CORRECT_CONCURRENCY on $(hostname)"
    exec python "$HERE/bench.py" "$IMPL" correct "$SETUP"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
