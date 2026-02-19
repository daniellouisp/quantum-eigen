#!/usr/bin/env python3
import argparse
import numpy as np

from scipy.linalg import eigh
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


def idx(i: int, j: int, N: int) -> int:
    """Map 2D grid index (i,j) to 1D flattened index."""
    return i * N + j


def potential_value(i: int, j: int, N: int, potential: str, dx: float) -> float:
    """Return V(i,j) for supported potentials."""
    if potential == "well":
        return 0.0

    # Coordinates centered at grid midpoint
    x = (i - (N - 1) / 2.0) * dx
    y = (j - (N - 1) / 2.0) * dx

    if potential == "harmonic":
        # isotropic
        return 4.0 * (x * x + y * y)

    if potential == "harmonic_aniso":
        # anisotropic: tighter in y than x (arbitrary example)
        kx, ky = 2.0, 8.0
        return kx * (x * x) + ky * (y * y)

    raise ValueError(f"Unknown potential: {potential}")


def apply_dirichlet0_bc_dense(H: np.ndarray, N: int) -> None:
    """
    Enforce homogeneous Dirichlet BC (psi=0 on boundary) by row replacement.
    For each boundary node k: set H[k,:]=0; H[k,k]=1.
    """
    for i in range(N):
        for j in range(N):
            is_boundary = (i == 0) or (i == N - 1) or (j == 0) or (j == N - 1)
            if is_boundary:
                k = idx(i, j, N)
                H[k, :] = 0.0
                H[k, k] = 1.0


def apply_dirichlet0_bc_sparse(H_lil: lil_matrix, N: int) -> None:
    """Same as apply_dirichlet0_bc_dense but for LIL sparse matrix."""
    for i in range(N):
        for j in range(N):
            is_boundary = (i == 0) or (i == N - 1) or (j == 0) or (j == N - 1)
            if is_boundary:
                k = idx(i, j, N)
                H_lil.rows[k] = [k]
                H_lil.data[k] = [1.0]


def build_hamiltonian_dense(N: int, potential: str, bc: str) -> np.ndarray:
    """
    Dense 2D Hamiltonian on N x N grid using 5-point stencil.

    Uses dx = 1/N so inv_dx2 = N^2 (arbitrary units).
    """
    dx = 1.0 / float(N)
    inv_dx2 = float(N * N)

    H = np.zeros((N * N, N * N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            row = idx(i, j, N)
            V = potential_value(i, j, N, potential, dx)

            # 2D Laplacian stencil:
            # diagonal: -4/dx^2 + V
            # neighbors: +1/dx^2
            H[row, row] = -4.0 * inv_dx2 + V

            if i > 0:
                H[row, idx(i - 1, j, N)] = inv_dx2
            if i < N - 1:
                H[row, idx(i + 1, j, N)] = inv_dx2
            if j > 0:
                H[row, idx(i, j - 1, N)] = inv_dx2
            if j < N - 1:
                H[row, idx(i, j + 1, N)] = inv_dx2

    if bc == "dirichlet0":
        apply_dirichlet0_bc_dense(H, N)

    return H


def build_hamiltonian_sparse(N: int, potential: str, bc: str):
    """
    Sparse 2D Hamiltonian (CSR) on N x N grid using 5-point stencil.
    """
    dx = 1.0 / float(N)
    inv_dx2 = float(N * N)

    H = lil_matrix((N * N, N * N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            row = idx(i, j, N)
            V = potential_value(i, j, N, potential, dx)

            H[row, row] = -4.0 * inv_dx2 + V

            if i > 0:
                H[row, idx(i - 1, j, N)] = inv_dx2
            if i < N - 1:
                H[row, idx(i + 1, j, N)] = inv_dx2
            if j > 0:
                H[row, idx(i, j - 1, N)] = inv_dx2
            if j < N - 1:
                H[row, idx(i, j + 1, N)] = inv_dx2

    if bc == "dirichlet0":
        apply_dirichlet0_bc_sparse(H, N)

    return H.tocsr()


def solve_eigen(
    N: int,
    potential: str,
    neigs: int,
    sparse: bool,
    bc: str,
):
    """
    Return the lowest `neigs` eigenpairs.

    Dense: full diagonalization (eigh) then slice.
    Sparse: eigsh (k=neigs, which="SA") for lowest algebraic eigenvalues.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if neigs <= 0 or neigs > N * N:
        raise ValueError("neigs must be in [1, N^2].")
    if potential not in {"well", "harmonic", "harmonic_aniso"}:
        raise ValueError("Invalid potential.")
    if bc not in {"none", "dirichlet0"}:
        raise ValueError("Invalid bc. Use 'none' or 'dirichlet0'.")

    if not sparse:
        H = build_hamiltonian_dense(N, potential, bc)
        vals, vecs = eigh(H)  # full spectrum
        order = np.argsort(vals)
        vals = vals[order][:neigs]
        vecs = vecs[:, order][:, :neigs]
        return vals, vecs

    # sparse path
    H = build_hamiltonian_sparse(N, potential, bc)

    # eigsh needs k < n for many cases; here n = N^2, k = neigs (small).
    # "SA" = smallest algebraic
    vals, vecs = eigsh(H, k=neigs, which="SA")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    return vals, vecs


def main():
    parser = argparse.ArgumentParser(description="2D Hamiltonian eigenvalue solver (finite-difference).")
    parser.add_argument("-N", type=int, default=10, help="Grid size per dimension (positive integer).")
    parser.add_argument(
        "--potential",
        type=str,
        default="well",
        choices=["well", "harmonic", "harmonic_aniso"],
        help="Potential type.",
    )
    parser.add_argument(
        "--neigs",
        type=int,
        default=5,
        help="Number of lowest eigenvalues to compute/print (<= N^2).",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse matrix + eigsh (recommended for larger N).",
    )
    parser.add_argument(
        "--bc",
        type=str,
        default="none",
        choices=["none", "dirichlet0"],
        help="Boundary conditions: none (default) or dirichlet0 (psi=0 at edges).",
    )
    parser.add_argument(
        "--save-psi",
        action="store_true",
        help="Save ground-state probability density |psi|^2 to a .npy file.",
    )
    parser.add_argument(
        "--psi-out",
        type=str,
        default=None,
        help="Output filename for |psi|^2 .npy (default: psi2_<potential>_N<N>.npy).",
    )
    args = parser.parse_args()

    vals, vecs = solve_eigen(
        N=args.N,
        potential=args.potential,
        neigs=args.neigs,
        sparse=args.sparse,
        bc=args.bc,
    )

    # Save eigenvalues
    eig_out = f"eigs_{args.potential}_N{args.N}.txt"
    with open(eig_out, "w") as f:
        np.savetxt(f, vals)

    print(f"N={args.N}, potential={args.potential}, neigs={args.neigs}, sparse={args.sparse}, bc={args.bc}")
    print(f"Wrote eigenvalues to: {eig_out}")
    print("Lowest eigenvalues:")
    for k, v in enumerate(vals, start=1):
        print(f"{k:2d}: {v:.8f}")

    # Optionally save ground-state probability density
    if args.save_psi:
        psi0 = vecs[:, 0]  # length N^2
        psi0_grid = psi0.reshape(args.N, args.N)
        prob = np.abs(psi0_grid) ** 2

        psi_out = args.psi_out or f"psi2_{args.potential}_N{args.N}.npy"
        np.save(psi_out, prob)
        print(f"Saved |psi0|^2 to: {psi_out}")


if __name__ == "__main__":
    main()

