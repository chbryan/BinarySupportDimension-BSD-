# Binary Support Dimension (BSD)

A minimal, runnable reference implementation of **Binary Support Dimension** — a 1/0 “dimension bit” for quantum states.

BSD collapses the **support dimension** of a density matrix \(\rho\) into a bit:

- **BSD(\(\rho\)) = 1** iff \(\rho\) is **rank-1** (pure state)
- **BSD(\(\rho\)) = 0** iff \(\rho\) has **rank > 1** (mixed state)

Equivalent characterizations (exact math; numerical code uses tolerance):

\[
\mathrm{BSD}(\rho)=1 \iff \rho^2=\rho \iff \mathrm{Tr}(\rho^2)=1
\]

## Files

- `bsd.py` (you create it from the code below): core functions + demos
- This README: concept + quick start

## Requirements

- Python 3.10+
- NumPy

Install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy
```

## Quick start

1) Create `bsd.py` with the code below.  
2) Run:

```bash
python bsd.py
```

You should see:
- Pure states report **BSD = 1** and purity ~ **1**
- Mixed states report **BSD = 0** and purity **< 1**
- Bell state is pure globally (BSD=1) but its reduced single-qubit states are mixed (BSD=0)

## `bsd.py`

```python
import numpy as np

def dagger(A: np.ndarray) -> np.ndarray:
    return A.conj().T

def is_hermitian(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.linalg.norm(A - dagger(A)) <= tol

def eigvals_hermitian(A: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(A)

def is_psd(A: np.ndarray, tol: float = 1e-10) -> bool:
    ev = eigvals_hermitian((A + dagger(A)) / 2)
    return np.min(ev) >= -tol

def trace_one(A: np.ndarray, tol: float = 1e-10) -> bool:
    return abs(np.trace(A) - 1.0) <= tol

def is_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> bool:
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        return False
    if not is_hermitian(rho, tol=tol):
        return False
    if not trace_one(rho, tol=1e-8):
        return False
    if not is_psd(rho, tol=1e-8):
        return False
    return True

def purity(rho: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=complex)
    return float(np.real(np.trace(rho @ rho)))

def rank_psd(rho: np.ndarray, tol: float = 1e-10) -> int:
    ev = eigvals_hermitian((rho + dagger(rho)) / 2)
    return int(np.sum(ev > tol))

def bsd(rho: np.ndarray, tol: float = 1e-10) -> int:
    """Binary Support Dimension: 1 iff rank(rho)==1 else 0."""
    rho = np.asarray(rho, dtype=complex)
    if not is_density_matrix(rho):
        raise ValueError("Input is not a valid density matrix.")
    return 1 if rank_psd(rho, tol=tol) == 1 else 0

def ketbra(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).reshape(-1, 1)
    n = np.linalg.norm(psi)
    if n == 0:
        raise ValueError("Zero vector is not a valid state.")
    psi = psi / n
    return psi @ dagger(psi)

def partial_trace_two_qubits(rho_4x4: np.ndarray, keep: str) -> np.ndarray:
    """Partial trace for 2 qubits: 4x4 -> 2x2."""
    rho = np.asarray(rho_4x4, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("Expected a 4x4 density matrix.")
    R = rho.reshape(2, 2, 2, 2)  # indices: a,b,a',b'
    keep = keep.upper()
    if keep == "A":
        return np.einsum("ab a'b->aa'", R)   # trace out b
    if keep == "B":
        return np.einsum("ab a b'->bb'", R)  # trace out a
    raise ValueError('keep must be "A" or "B".')

def report(name: str, rho: np.ndarray):
    print(name)
    print("  valid density:", is_density_matrix(rho))
    print("  purity:", purity(rho))
    print("  rank:", rank_psd(rho))
    print("  BSD:", bsd(rho))
    print()

if __name__ == "__main__":
    # Pure |0>
    ket0 = np.array([1, 0], dtype=complex)
    rho0 = ketbra(ket0)
    report("Pure |0><0|", rho0)

    # Maximally mixed I/2
    rho_mm = np.eye(2, dtype=complex) / 2
    report("Maximally mixed I/2", rho_mm)

    # Bell state |Φ+> = (|00> + |11>)/sqrt(2)
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho_bell = ketbra(phi_plus)
    report("Bell state |Φ+><Φ+| (4x4)", rho_bell)

    # Reduced states of Bell state (mixed locally)
    rho_A = partial_trace_two_qubits(rho_bell, keep="A")
    rho_B = partial_trace_two_qubits(rho_bell, keep="B")
    report("Reduced state rho_A (2x2)", rho_A)
    report("Reduced state rho_B (2x2)", rho_B)
```

## Notes

- BSD is a **binary invariant** of a density matrix’s support: it does **not** claim physical spacetime has dimension 0 or 1.
- Numerical computations use tolerances; purity may print as `0.999999999999` for pure states depending on floating-point noise.
