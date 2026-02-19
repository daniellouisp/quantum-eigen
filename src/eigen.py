#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.linalg import eigh


def build_2d_hamiltonian(N: int = 20, potential: str = "well") -> np.ndarray:
    """
    Build a discretized 2D Hamiltonian on an N x N grid using a 5-point stencil.

    H ~ Laplacian + V(x,y)

    Returns
    -------
    H : ndarray, shape (N^2, N^2)
        Real symmetric Hamiltonian matrix.
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if potential not in {"well", "harmonic"}:
        raise ValueError("potential must be one of: 'well', 'harmonic'.")

    dx = 1.0 / float(N)
    inv_dx2 = float(N * N)  # 1/dx^2 since dx = 1/N

    H = np.zeros((N * N, N * N), dtype=np.float64)

    def idx(i: int, j: int) -> int:
        return i * N + j

    def V(i: int, j: int) -> float:
        if potential == "well":
            return 0.0
        # harmonic
        x = (i - N / 2.0) * dx
        y = (j - N / 2.0) * dx
        return 4.0 * (x**2 + y**2)

    # Build with the standard 2D Laplacian stencil:
    # diagonal: -4/dx^2 + V
    # neighbors: +1/dx^2
    for i in range(N):
        for j in range(N):
            row = idx(i, j)
            H[row, row] = -4.0 * inv_dx2 + V(i, j)

            if i > 0:
                H[row, idx(i - 1, j)] = inv_dx2
            if i < N - 1:
                H[row, idx(i + 1, j)] = inv_dx2
            if j > 0:
                H[row, idx(i, j - 1)] = inv_dx2
            if j < N - 1:
                H[row, idx(i, j + 1)] = inv_dx2

    return H


def solve_eigen(N: int = 20, potential: str = "well", n_eigs: int | None = None):
    """
    Build 2D Hamiltonian and solve eigenproblem.

    Parameters
    ----------
    N : int
        Grid points per dimension.
    potential : str
        'well' or 'harmonic'
    n_eigs : int or None
        Number of lowest eigenvalues to return (<= N^2). If None, returns all.

    Returns
    -------
    vals_sorted : ndarray
    vecs_sorted : ndarray
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if potential not in {"well", "harmonic"}:
        raise ValueError("potential must be one of: 'well', 'harmonic'.")
    if n_eigs is not None:
        if not isinstance(n_eigs, int) or n_eigs <= 0:
            raise ValueError("n_eigs must be a positive integer or None.")
        if n_eigs > N * N:
            raise ValueError("n_eigs must be <= N^2.")

    H = build_2d_hamiltonian(N, potential)

    # Full dense eigen-solve (fine for small N; grows fast with N^2!)
    vals, vecs = eigh(H)

    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]

    if n_eigs is None:
        return vals_sorted, vecs_sorted
    return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]


def main():
    parser = argparse.ArgumentParser(description="2D Hamiltonian eigenvalue solver (finite-difference).")
    parser.add_argument("-N", type=int, default=10, help="Grid size per dimension (positive integer).")
    parser.add_argument(
        "--potential",
        type=str,
        default="well",
        choices=["well", "harmonic"],
        help="Potential type.",
    )
    parser.add_argument(
        "--neigs",
        type=int,
        default=5,
        help="Number of lowest eigenvalues to print (positive int, <= N^2).",
    )
    args = parser.parse_args()

    vals, _ = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.neigs)

    # NEW: save eigenvalues to a text file
    outname = f"eigs_{args.potential}_N{args.N}.txt"
    with open(outname, "w") as f:
        np.savetxt(f, vals)

    print(f"N={args.N}, potential={args.potential}, neigs={args.neigs}")
    print(f"Wrote eigenvalues to: {outname}")
    print("Lowest eigenvalues:")
    for k, v in enumerate(vals, start=1):
        print(f"{k:2d}: {v:.8f}")


if __name__ == "__main__":
    main()

