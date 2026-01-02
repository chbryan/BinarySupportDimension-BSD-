import numpy as np

def dagger(A):
    return A.conj().T

def density_from_state(psi):
    """|psi><psi| normalized."""
    psi = np.asarray(psi, dtype=complex).reshape(-1, 1)
    nrm = np.linalg.norm(psi)
    if nrm == 0:
        raise ValueError("Zero vector is not a valid quantum state.")
    psi = psi / nrm
    return psi @ dagger(psi)

def is_density_matrix(rho, tol=1e-10):
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        return False
    if np.linalg.norm(rho - dagger(rho)) > tol:
        return False
    tr = np.trace(rho)
    if abs(tr - 1.0) > 1e-8:
        return False
    # PSD check via eigenvalues
    evals = np.linalg.eigvalsh(rho)
    return np.min(evals) >= -1e-8

def purity(rho):
    rho = np.asarray(rho, dtype=complex)
    return float(np.real(np.trace(rho @ rho)))

def bqd(rho, tol=1e-10):
    """
    Binary Quantum Dimension:
    1 iff rho^2 = rho (pure / rank-1),
    0 otherwise.
    """
    rho = np.asarray(rho, dtype=complex)
    if not is_density_matrix(rho):
        raise ValueError("Input is not a valid density matrix.")
    p = purity(rho)
    # Numerical tolerance: purity should be 1 for pure states, <1 for mixed
    return 1 if abs(p - 1.0) <= 1e-8 else 0

def random_pure_state(dim, rng=np.random.default_rng()):
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return v

def random_mixed_state(dim, k=None, rng=np.random.default_rng()):
    """
    Create a mixed state as a convex mixture of k pure states.
    """
    if k is None:
        k = min(dim, 4)
    weights = rng.random(k)
    weights = weights / np.sum(weights)
    rho = np.zeros((dim, dim), dtype=complex)
    for w in weights:
        psi = random_pure_state(dim, rng)
        rho += w * density_from_state(psi)
    return rho

def partial_trace_two_qubits(rho, keep="A"):
    """
    Partial trace for a 2-qubit density matrix (4x4).
    keep="A" traces out B, returns rho_A (2x2)
    keep="B" traces out A, returns rho_B (2x2)
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("Expected a 4x4 density matrix for two qubits.")
    # Reshape indices: (a,b; a',b') -> tensor rho[a,b,a',b']
    R = rho.reshape(2, 2, 2, 2)
    if keep.upper() == "A":
        # trace over b: sum_b R[a,b,a',b]
        return np.einsum("ab a'b->aa'", R)
    elif keep.upper() == "B":
        # trace over a: sum_a R[a,b,a,b']
        return np.einsum("ab a b'->bb'", R)
    else:
        raise ValueError('keep must be "A" or "B".')

# --- Demonstrations ---

# 1) Pure single-qubit state |0>
ket0 = np.array([1, 0], dtype=complex)
rho0 = density_from_state(ket0)
print("Pure |0> purity:", purity(rho0), "BQD:", bqd(rho0))

# 2) Maximally mixed single-qubit state I/2
rho_mm = np.eye(2, dtype=complex) / 2
print("Maximally mixed purity:", purity(rho_mm), "BQD:", bqd(rho_mm))

# 3) Two-qubit Bell state |Φ+> = (|00> + |11>)/sqrt(2)
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = density_from_state(phi_plus)
print("Bell state purity:", purity(rho_bell), "BQD:", bqd(rho_bell))

# 4) Reduced state of Bell state (should be maximally mixed on each subsystem)
rho_A = partial_trace_two_qubits(rho_bell, keep="A")
rho_B = partial_trace_two_qubits(rho_bell, keep="B")
print("Reduced A purity:", purity(rho_A), "BQD:", bqd(rho_A))
print("Reduced B purity:", purity(rho_B), "BQD:", bqd(rho_B))

# 5) Random examples
rng = np.random.default_rng(0)
psi = random_pure_state(5, rng)
rho_p = density_from_state(psi)
rho_m = random_mixed_state(5, k=3, rng=rng)

print("Random pure (dim=5) purity:", purity(rho_p), "BQD:", bqd(rho_p))
print("Random mixed (dim=5) purity:", purity(rho_m), "BQD:", bqd(rho_m))
