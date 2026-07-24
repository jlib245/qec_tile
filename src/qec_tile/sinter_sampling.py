"""Parallel shot collection through sinter.

Wraps ``sinter.collect`` so a benchmark can decode many circuits at once
across worker processes, instead of the one-shot-at-a-time loop in
``circuit.circuit_failure_rate``.  Both paths use the same BP+OSD settings.

Worker scheduling is nondeterministic, so counts differ from run to run —
when a reproducible (seeded) number is needed, use the serial
``circuit_failure_rate`` instead.
"""
from __future__ import annotations

import sinter
import stim
from ldpc import SinterBpOsdDecoder


def _bposd() -> SinterBpOsdDecoder:
    # Mirror decode.make_decoder so serial and parallel results are comparable.
    return SinterBpOsdDecoder(
        bp_method="minimum_sum",
        max_iter=50,
        osd_method="osd_cs",
        osd_order=7,
    )


# A future NN decoder registers a sinter.Decoder factory here, next to its
# entry in decode.DECODERS.  Factories, not instances: sinter pickles the
# decoder into each worker, and a fresh object per collect() avoids any
# shared state between runs.
SINTER_DECODERS = {"bposd": _bposd}


def collect(circuits: dict[object, stim.Circuit], decoder: str,
            max_shots: int, max_errors: int | None = None,
            workers: int = 8,
            progress: bool = True) -> dict[object, tuple[int, int]]:
    """Decode every circuit in parallel -> ``{key: (shots, errors)}``.

    Tasks carry an index in their metadata because sinter returns stats in
    completion order and JSON would mangle tuple keys.
    """
    build = SINTER_DECODERS.get(decoder)
    if build is None:
        raise ValueError(
            f"unknown decoder {decoder!r}; have {sorted(SINTER_DECODERS)}")

    # A circuit whose DEM has no error mechanism can never fail, and the
    # decoder hangs on the empty problem — answer trivially instead.
    results: dict = {}
    noisy: dict = {}
    for key, circuit in circuits.items():
        dem = circuit.detector_error_model()
        if any(inst.type == "error" for inst in dem.flattened()):
            noisy[key] = circuit
        else:
            results[key] = (max_shots, 0)
    if not noisy:
        return results

    keys = list(noisy)
    tasks = [sinter.Task(circuit=noisy[key], json_metadata={"index": i})
             for i, key in enumerate(keys)]
    task_stats = sinter.collect(
        num_workers=workers,
        tasks=tasks,
        decoders=[decoder],
        custom_decoders={decoder: build()},
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=progress,     # stderr, so CSV/stdout stay clean
    )
    results.update({keys[stat.json_metadata["index"]]:
                    (stat.shots, stat.errors) for stat in task_stats})
    return results
