from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .operators import NEIGHBOR_OFFSETS, CompassOperator, get_operator

__all__ = [
    "QCEDStatevectorResult",
    "QCEDShotResult",
    "QPUUsageLimitError",
    "run_qced",
    "run_qced_batch",
    "neighbors_from_patch",
    "BACKENDS",
]

_M = 8
_DATA_WIRES = [0, 1, 2]
_ANCILLA = 3


class QPUUsageLimitError(RuntimeError):
    """Raised when the IBM instance has run out of runtime (usage limit) or a
    submission fails, so the QPU run stops cleanly. Everything completed so
    far is already in the checkpoint; re-running the same command resumes."""


# substrings that mark an "out of runtime / quota" condition in the warning
# or error text emitted by qiskit-ibm-runtime
_USAGE_LIMIT_HINTS = (
    "usage limit", "met its usage limit", "will not run",
    "no run time", "run time left", "insufficient", "quota", "exhaust",
)


def _is_usage_limit(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in _USAGE_LIMIT_HINTS)


def _require_pennylane():
    """Import PennyLane, raising an informative error if missing."""
    try:
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PennyLane is required for the quantum submodule. "
            "Install with: pip install 'compass-gradient[quantum]'"
        ) from exc
    return qml


def neighbors_from_patch(patch: np.ndarray) -> np.ndarray:
    """Extract the 8-neighbor intensity vector of a 3x3 patch in the
    counterclockwise enumeration used by the classical package."""
    patch = np.asarray(patch, dtype=np.float64)
    if patch.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 patch, got shape {patch.shape}.")
    return np.array([patch[1 + dr, 1 + dc] for dr, dc in NEIGHBOR_OFFSETS])


def _bits_msb(k: int) -> "list[int]":
    """3-bit binary expansion of k, most significant bit first."""
    return [(k >> 2) & 1, (k >> 1) & 1, k & 1]


def _apply_u_d(r: np.ndarray, lam: float, qml) -> None:
    """Apply the U_D block: for each basis state |k>, a controlled
    PhaseShift of angle arg r(k) on the data register and a controlled
    RY of angle 2*arcsin(|r(k)|/lam) on the ancilla. Identity gates
    are skipped."""
    r_tilde = r / lam
    tol = 1e-12
    for k in range(_M):
        mag = float(np.abs(r_tilde[k]))
        phi = float(np.angle(r[k])) if mag > tol else 0.0
        bits = _bits_msb(k)

        if abs(phi) > tol:
            if k == 0:
                qml.ctrl(
                    qml.GlobalPhase(-phi, wires=_ANCILLA),
                    control=_DATA_WIRES,
                    control_values=bits,
                )
            else:
                target_pos = bits.index(1)
                target = _DATA_WIRES[target_pos]
                controls = [w for i, w in enumerate(_DATA_WIRES) if i != target_pos]
                cvals = [b for i, b in enumerate(bits) if i != target_pos]
                qml.ctrl(
                    qml.PhaseShift(phi, wires=target),
                    control=controls,
                    control_values=cvals,
                )

        theta = 2.0 * float(np.arcsin(np.clip(mag, 0.0, 1.0)))
        if theta > tol:
            qml.ctrl(
                qml.RY(theta, wires=_ANCILLA),
                control=_DATA_WIRES,
                control_values=bits,
            )


def _pipeline(p_norm: np.ndarray, r: np.ndarray, lam: float, qml) -> None:
    """Apply the full QCED gate sequence without measurement."""
    qml.StatePrep(p_norm, wires=_DATA_WIRES)

    qml.adjoint(qml.QFT)(wires=_DATA_WIRES)
    #qml.QFT(wires=_DATA_WIRES)
    _apply_u_d(r, lam, qml)
    qml.QFT(wires=_DATA_WIRES)
    #qml.adjoint(qml.QFT)(wires=_DATA_WIRES)


def _prepare_batch(patches, remove_dc: bool):
    """Convert a sequence of 3x3 patches or length-8 vectors into
    normalized state vectors; returns (states, norms, active_mask)
    where inactive entries are degenerate (zero-norm) inputs."""
    vecs = []
    for p in patches:
        p = np.asarray(p, dtype=np.float64)
        v = neighbors_from_patch(p) if p.shape == (3, 3) else p
        if v.shape != (_M,):
            raise ValueError(f"Expected 8 neighbor values, got {v.shape}.")
        if remove_dc:
            v = v - v.mean()
        vecs.append(v)
    V = np.stack(vecs)
    norms = np.linalg.norm(V, axis=1)
    active = norms > 1e-12
    P = np.zeros_like(V)
    P[active] = V[active] / norms[active, None]
    return P, norms, active


@dataclass(frozen=True)
class QCEDStatevectorResult:
    """Exact adopted-block (ancilla = |1>) amplitudes from the
    statevector backend."""

    amplitudes: np.ndarray
    scale: float
    input_norm: float

    @property
    def edge_values(self) -> np.ndarray:
        """Signed edge values on the classical scale."""
        return self.amplitudes.real * self.scale * self.input_norm

    @property
    def success_probability(self) -> float:
        """Probability of measuring the ancilla in |1>."""
        return float(np.sum(np.abs(self.amplitudes) ** 2))

    @property
    def probs(self) -> np.ndarray:
        """Post-selected probability distribution over directions."""
        p = np.abs(self.amplitudes) ** 2
        s = p.sum()
        return p / s if s > 0 else p

    @property
    def direction(self) -> int:
        """Direction index with the largest post-selected probability,
        or -1 when the edge response is identically zero."""
        if self.success_probability == 0.0:
            return -1
        return int(np.argmax(np.abs(self.amplitudes)))


@dataclass(frozen=True)
class QCEDShotResult:
    """Post-selected measurement statistics from a finite-shot
    backend."""

    counts: np.ndarray
    shots: int
    valid_shots: int
    scale: float
    input_norm: float

    @property
    def success_probability(self) -> float:
        """Estimated probability of measuring the ancilla in |1>."""
        return self.valid_shots / self.shots if self.shots else 0.0

    @property
    def probs(self) -> np.ndarray:
        """Estimated post-selected probability distribution."""
        if self.valid_shots == 0:
            return np.zeros(_M)
        return self.counts / self.valid_shots

    @property
    def direction(self) -> int:
        """Direction index with the largest post-selected count, or -1
        when no shot survived post-selection."""
        if self.valid_shots == 0:
            return -1
        return int(np.argmax(self.counts))

    @property
    def edge_magnitudes(self) -> np.ndarray:
        """Estimated |EG_m| on the classical scale (signs are not
        recoverable from measurement statistics)."""
        return self.scale * self.input_norm * np.sqrt(self.counts / self.shots)


def _zero_shot_result(shots: int, lam: float) -> QCEDShotResult:
    """Zero-count result for a degenerate (constant-neighbor) input."""
    return QCEDShotResult(
        counts=np.zeros(_M, dtype=np.int64), shots=shots,
        valid_shots=0, scale=lam, input_norm=0.0)


def _sample_counts(probs16: np.ndarray, shots: int, rng) -> "tuple[np.ndarray, int]":
    """Draw a multinomial shot histogram from a 16-outcome
    distribution and post-select the ancilla |1> (odd) outcomes."""
    p = np.clip(probs16.real, 0.0, None)
    p = p / p.sum()
    c16 = rng.multinomial(shots, p)
    counts = c16[1::2].astype(np.int64)
    return counts, int(counts.sum())


def _broadcast_states(P_active: np.ndarray, r, lam, qml) -> np.ndarray:
    """Execute the circuit for all active inputs in one broadcast call
    and return the (N, 16) statevector array."""
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(p_batch):
        _pipeline(p_batch, r, lam, qml)
        return qml.state()

    return np.atleast_2d(np.asarray(circuit(P_active)))


def _run_statevector(P, norms, active, r, lam, qml, **_):
    """Statevector backend: one broadcast execution for the whole
    batch; returns exact adopted-block amplitudes."""
    results = [None] * len(P)
    idx = np.flatnonzero(active)
    if idx.size:
        states = _broadcast_states(P[idx], r, lam, qml)
        for n, i in enumerate(idx):
            results[i] = QCEDStatevectorResult(
                amplitudes=states[n, 1::2], scale=lam,
                input_norm=float(norms[i]))
    for i in np.flatnonzero(~active):
        results[i] = QCEDStatevectorResult(
            amplitudes=np.zeros(_M, dtype=complex), scale=lam,
            input_norm=0.0)
    return results


def _run_shots(P, norms, active, r, lam, qml, shots=1024, seed=None, **_):
    """Noiseless shot backend: exact outcome distributions from one
    broadcast execution, sampled with a classical multinomial sampler
    (statistically identical to circuit-level sampling)."""
    results = [None] * len(P)
    idx = np.flatnonzero(active)
    if idx.size:
        states = _broadcast_states(P[idx], r, lam, qml)
        probs = np.abs(states) ** 2
        rng = np.random.default_rng(seed)
        for n, i in enumerate(idx):
            counts, valid = _sample_counts(probs[n], shots, rng)
            results[i] = QCEDShotResult(
                counts=counts, shots=shots, valid_shots=valid,
                scale=lam, input_norm=float(norms[i]))
    for i in np.flatnonzero(~active):
        results[i] = _zero_shot_result(shots, lam)
    return results


def _noisy_ops(p_norm: np.ndarray, r, lam, qml, p1: float, p2: float):
    """Build the pipeline operation list with a depolarizing channel of
    probability p1 (single-qubit gates) or p2 (multi-qubit gates)
    inserted on every wire after each operation."""
    with qml.queuing.AnnotatedQueue() as q:
        _pipeline(p_norm, r, lam, qml)
    ops = []
    for op in q.queue:
        ops.append(op)
        p = p1 if len(op.wires) == 1 else p2
        if p > 0:
            for w in op.wires:
                ops.append(qml.DepolarizingChannel(p, wires=w))
    return ops


def _run_noisy(P, norms, active, r, lam, qml, shots=1024, seed=None,
               noise_p1=3.14e-4, noise_p2=2.79e-3, readout=9.89e-3, **_):
    """Noisy shot backend: exact noisy outcome distributions from a
    density-matrix simulation (one batched submission of all circuits),
    sampled with a classical multinomial sampler."""
    results = [None] * len(P)
    idx = np.flatnonzero(active)
    if idx.size:
        dev = qml.device("default.mixed", wires=4)
        tapes = []
        for i in idx:
            ops = _noisy_ops(P[i], r, lam, qml, noise_p1, noise_p2)
            if readout > 0:
                for w in [*_DATA_WIRES, _ANCILLA]:
                    ops.append(qml.BitFlip(readout, wires=w))
            tapes.append(qml.tape.QuantumScript(
                ops, [qml.probs(wires=[*_DATA_WIRES, _ANCILLA])]))
        all_probs = qml.execute(tapes, dev)
        rng = np.random.default_rng(seed)
        for n, i in enumerate(idx):
            counts, valid = _sample_counts(
                np.asarray(all_probs[n]), shots, rng)
            results[i] = QCEDShotResult(
                counts=counts, shots=shots, valid_shots=valid,
                scale=lam, input_norm=float(norms[i]))
    for i in np.flatnonzero(~active):
        results[i] = _zero_shot_result(shots, lam)
    return results


def _run_qpu(P, norms, active, r, lam, qml, shots=1024, device=None,
             chunk_size=16, resume_state=None, on_chunk_done=None, **_):
    """QPU backend with optional checkpoint/resume.

    Submits one post-selected sampling circuit per input to a user-supplied
    hardware device, in batches of `chunk_size` circuits. After each chunk,
    if `on_chunk_done` is given, it is called with that chunk's per-pixel
    records -- a list of dicts {"index", "counts", "valid_shots",
    "input_norm"} -- so the caller can persist a checkpoint; the checkpoint
    granularity is therefore one chunk.

    `resume_state` ({index -> record}) lists pixels already completed in an
    earlier run; they are reconstructed and NOT resubmitted, so an interrupted
    run resumes where it left off. `index` is the position in `P` (i.e. in the
    batch passed to run_qced_batch). Degenerate (zero-norm) pixels need no
    circuit and are emitted as zero-count records so a checkpoint can reach
    every pixel. Returns the full results list (resumed + newly run +
    degenerate) in input order.

    With resume_state=None and on_chunk_done=None this is a plain chunked QPU
    run (the behaviour demo.py relies on)."""
    if device is None:
        raise ValueError(
            "The 'qpu' backend requires a constructed PennyLane device "
            "passed as device=..., e.g. "
            "qml.device('qiskit.remote', wires=4, backend=...)."
        )
    resume_state = {int(k): v for k, v in (resume_state or {}).items()}
    chunk_size = max(1, int(chunk_size))
    results = [None] * len(P)

    # 1) pixels already in the checkpoint: reconstruct, do not resubmit
    for i, rec in resume_state.items():
        if 0 <= i < len(P):
            results[i] = QCEDShotResult(
                counts=np.asarray(rec["counts"], dtype=np.int64), shots=shots,
                valid_shots=int(rec["valid_shots"]), scale=lam,
                input_norm=float(rec["input_norm"]))

    pending = []   # records awaiting the next on_chunk_done flush

    def _flush():
        if pending and on_chunk_done is not None:
            on_chunk_done(list(pending))
        pending.clear()

    # 2) degenerate (zero-norm) pixels: no circuit, zero counts
    for i in np.flatnonzero(~active):
        i = int(i)
        if i in resume_state:
            continue
        results[i] = _zero_shot_result(shots, lam)
        pending.append({"index": i, "counts": [0] * _M,
                        "valid_shots": 0, "input_norm": 0.0})

    # 3) active pixels not yet done: submit in chunks, checkpoint each chunk
    todo = [int(i) for i in np.flatnonzero(active) if int(i) not in resume_state]
    for c0 in range(0, len(todo), chunk_size):
        chunk = todo[c0:c0 + chunk_size]
        tapes = []
        for i in chunk:
            with qml.queuing.AnnotatedQueue() as q:
                _pipeline(P[i], r, lam, qml)
                qml.sample(wires=[*_DATA_WIRES, _ANCILLA])
            tapes.append(qml.tape.QuantumScript.from_queue(q, shots=shots))
        # Submit this chunk, capturing warnings so an IBM "usage limit"
        # notice can be turned into a clean stop. If the submission itself
        # raises (limit, auth, queue error), the previous chunks are already
        # checkpointed, so we stop cleanly and the run can be resumed.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                all_samples = qml.execute(tapes, device)
            except Exception as exc:
                _flush()
                if _is_usage_limit(str(exc)):
                    raise QPUUsageLimitError(
                        f"IBM instance out of runtime while submitting the "
                        f"chunk at pixel {chunk[0]} ({exc}). Checkpoint saved; "
                        f"add runtime and re-run the same command to resume."
                    ) from exc
                raise QPUUsageLimitError(
                    f"QPU submission failed at pixel {chunk[0]} ({exc}). "
                    f"Checkpoint saved; re-run the same command to resume."
                ) from exc
        # this chunk completed -> record and checkpoint it
        for n, i in enumerate(chunk):
            samples = np.asarray(all_samples[n])
            kept = samples[samples[:, 3] == 1]
            m_values = kept[:, 0] * 4 + kept[:, 1] * 2 + kept[:, 2]
            counts = np.bincount(m_values, minlength=_M).astype(np.int64)
            results[i] = QCEDShotResult(
                counts=counts, shots=shots, valid_shots=int(kept.shape[0]),
                scale=lam, input_norm=float(norms[i]))
            pending.append({"index": i, "counts": counts.tolist(),
                            "valid_shots": int(kept.shape[0]),
                            "input_norm": float(norms[i])})
        _flush()   # persist this chunk before submitting the next

        # re-emit any non-limit warnings (so they are not swallowed), and if a
        # usage-limit notice appeared, stop BEFORE submitting the next chunk
        # (which would not run) -- the chunk just saved is kept.
        limit_hit = None
        for wn in caught:
            if _is_usage_limit(str(wn.message)):
                limit_hit = str(wn.message)
            else:
                warnings.warn_explicit(wn.message, wn.category,
                                       wn.filename, wn.lineno)
        if limit_hit is not None:
            raise QPUUsageLimitError(
                "IBM instance usage limit reached (" + limit_hit.strip() +
                "). Stopped after the last completed chunk; the checkpoint is "
                "saved -- add runtime and re-run the same command to resume.")

    _flush()       # flush degenerate-only records when there were no chunks
    return results


BACKENDS = {
    "statevector": _run_statevector,
    "shots": _run_shots,
    "noisy": _run_noisy,
    "qpu": _run_qpu,
}


def run_qced_batch(
    patches,
    operator: "str | CompassOperator" = "sobel",
    backend: str = "statevector",
    remove_dc: bool = False,
    **backend_kwargs,
):
    """Run the QCED circuit on a sequence of 3x3 patches (or length-8
    neighbor vectors), submitting the whole batch to the selected
    backend at once; returns a list of results in input order.

    Backend-specific options: shots, seed ("shots", "noisy", "qpu");
    noise_p1, noise_p2, readout ("noisy"); device ("qpu")."""
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}; available: {sorted(BACKENDS)}"
        )
    qml = _require_pennylane()
    P, norms, active = _prepare_batch(patches, remove_dc)
    op = get_operator(operator)
    r = op.eigenvalues()
    lam = float(np.max(np.abs(r)))
    return BACKENDS[backend](P, norms, active, r, lam, qml,
                             **backend_kwargs)


def run_qced(
    patch: np.ndarray,
    operator: "str | CompassOperator" = "sobel",
    backend: str = "statevector",
    remove_dc: bool = False,
    **backend_kwargs,
):
    """Run the QCED circuit on a single patch (batch of size one)."""
    return run_qced_batch([patch], operator, backend, remove_dc,
                          **backend_kwargs)[0]