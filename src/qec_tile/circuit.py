"""Circuit-level memory-Z experiment for a tile code, built with stim.

One round measures every X stabilizer, then every Z stabilizer, each with its
own ancilla.  The CNOT schedule exploits translation invariance: a global
ordering of the tile's offsets is fixed, and at time slot ``o`` every check
touches the qubit at ``anchor + o``.  Since ``anchor + o`` determines the
anchor, two checks of the same type can never collide in a slot; truncated
boundary checks simply skip the slots they lost.  X and Z layers are kept in
disjoint slots (depth ~ 2w), trading depth for scheduling simplicity.

``memory_z_base`` emits the circuit noise-free with TICKs between moments;
noise is a post-processing step through ``noise_model.NoiseModel`` (vendored
Gidney implementation), which injects gate noise plus per-moment idle noise.
``memory_z_circuit`` keeps the simple uniform-``p`` model as a wrapper, and
``NoiseModel.SI1000(p)`` gives the superconducting-inspired canonical model.

Detectors: data start in |0>, so Z-check outcomes are deterministic in round
one and compared round-to-round afterwards; X-check outcomes only from round
two on; a final transversal Z readout reconstructs each Z check once more.
The k observables are the LZ rows applied to that final readout.
"""
from __future__ import annotations

import numpy as np
import stim
from ldpc.ckt_noise import detector_error_model_to_check_matrices

from .decode import DECODERS
from .noise_model import NoiseModel


def _schedule(H, anchors, qubits) -> tuple[list, list[dict]]:
    """Global time slots (tile offsets) and, per check, offset -> data column."""
    per_check: list[dict] = []
    slots = set()
    for row, (anchor_x, anchor_y) in zip(np.asarray(H), anchors):
        offsets = {}
        for col in np.flatnonzero(row):
            orient, x, y = qubits[col]
            offset = (orient, x - anchor_x, y - anchor_y)
            offsets[offset] = int(col)
            slots.add(offset)
        per_check.append(offsets)
    return sorted(slots), per_check


def memory_z_base(code, rounds: int) -> stim.Circuit:
    """Noise-free memory-Z circuit for ``rounds`` syndrome rounds.

    Moments are separated by TICKs so a ``noise_model.NoiseModel`` can inject
    gate noise and, crucially, idle noise per moment as a post-processing step.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    n = code.n
    mx, mz = code.HX.shape[0], code.HZ.shape[0]
    data = list(range(n))
    x_anc = [n + i for i in range(mx)]
    z_anc = [n + mx + j for j in range(mz)]
    x_slots, x_checks = _schedule(code.HX, code.x_anchors, code.qubits)
    z_slots, z_checks = _schedule(code.HZ, code.z_anchors, code.qubits)
    _, LZ = code.logicals()

    circuit = stim.Circuit()
    # Coordinates make stim's timeslice diagrams draw the actual lattice:
    # data qubits at edge midpoints, ancillas inside their anchor's box
    # (0.25/0.75 offsets keep bulk X and Z ancillas apart).
    for col, (orient, x, y) in enumerate(code.qubits):
        xy = (x + 0.5, y) if orient == "H" else (x, y + 0.5)
        circuit.append("QUBIT_COORDS", [col], xy)
    for i, (anchor_x, anchor_y) in enumerate(code.x_anchors):
        circuit.append("QUBIT_COORDS", [x_anc[i]],
                       (anchor_x + 0.25, anchor_y + 0.25))
    for j, (anchor_x, anchor_y) in enumerate(code.z_anchors):
        circuit.append("QUBIT_COORDS", [z_anc[j]],
                       (anchor_x + 0.75, anchor_y + 0.75))

    circuit.append("R", data + z_anc)          # |0> data and Z ancillas
    circuit.append("RX", x_anc)                # |+> X ancillas
    circuit.append("TICK")

    for round_index in range(rounds):
        # X layer: ancilla is the control (measures X on its support).
        for slot in x_slots:
            pairs = [q for i, offsets in enumerate(x_checks)
                     if slot in offsets for q in (x_anc[i], offsets[slot])]
            circuit.append("CX", pairs)
            circuit.append("TICK")
        # Z layer: data is the control.
        for slot in z_slots:
            pairs = [q for j, offsets in enumerate(z_checks)
                     if slot in offsets for q in (offsets[slot], z_anc[j])]
            circuit.append("CX", pairs)
            circuit.append("TICK")

        # Measure and reset the ancillas (mx results, then mz).
        circuit.append("MRX", x_anc)
        circuit.append("MR", z_anc)

        stride = mx + mz                       # measurements per round
        for j in range(mz):
            current = -(mz - j)
            if round_index == 0:               # deterministic against |0>
                circuit.append("DETECTOR", [stim.target_rec(current)])
            else:
                circuit.append("DETECTOR", [stim.target_rec(current),
                                            stim.target_rec(current - stride)])
        if round_index > 0:
            for i in range(mx):
                current = -(mz + mx - i)
                circuit.append("DETECTOR", [stim.target_rec(current),
                                            stim.target_rec(current - stride)])
        circuit.append("TICK")

    # Final transversal Z readout of the data reconstructs each Z check.
    circuit.append("M", data)
    for j in range(mz):
        targets = [stim.target_rec(-(n - col))
                   for col in np.flatnonzero(code.HZ[j])]
        targets.append(stim.target_rec(-(n + mz - j)))
        circuit.append("DETECTOR", targets)
    for l, logical in enumerate(LZ):
        targets = [stim.target_rec(-(n - col))
                   for col in np.flatnonzero(logical)]
        circuit.append("OBSERVABLE_INCLUDE", targets, l)
    return circuit


def memory_z_circuit(code, rounds: int, p: float) -> stim.Circuit:
    """Memory-Z circuit under uniform noise: every gate, measurement and reset
    fails with probability ``p``, no idle noise.

    Kept for the "circuit" benchmark axis; the canonical alternative is
    ``NoiseModel.SI1000(p).noisy_circuit(memory_z_base(code, rounds))``.
    """
    uniform = NoiseModel(
        idle=0.0,
        measure_reset_idle=0.0,
        noisy_gates={"CX": p, "R": p, "RX": p, "M": p, "MR": p, "MRX": p},
    )
    return uniform.noisy_circuit(memory_z_base(code, rounds))


def circuit_failure_rate(circuit: stim.Circuit, shots: int, decoder: str,
                         seed: int | None = None) -> float:
    """Sample the circuit itself and decode its detection events.

    The decoder works on the detector error model's matrices, but the events
    come from stim sampling the actual circuit — the DEM is used as the
    decoder's map, not as the noise source.  A shot fails when the observable
    flips predicted from the decoded error disagree with the sampled ones
    (equivalent to the residual test, without needing the true error).
    """
    dem = circuit.detector_error_model()
    matrices = detector_error_model_to_check_matrices(
        dem, allow_undecomposed_hyperedges=True)   # BP+OSD takes hyperedges
    if matrices.check_matrix.shape[1] == 0:        # noiseless: nothing to fail
        return 0.0

    build = DECODERS.get(decoder)
    if build is None:
        raise ValueError(
            f"unknown decoder {decoder!r}; have {sorted(DECODERS)}")
    decoder_obj = build(matrices.check_matrix, matrices.priors)

    sampler = circuit.compile_detector_sampler(seed=seed)
    detections, observed = sampler.sample(shots, separate_observables=True)

    A = matrices.observables_matrix.toarray().astype(np.uint8)
    failures = 0
    for detection, actual in zip(detections, observed):
        x_hat = decoder_obj.decode(detection.astype(np.uint8))
        predicted = (A @ x_hat) % 2
        failures += bool((predicted != actual).any())
    return failures / shots
