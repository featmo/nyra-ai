# Singular Value Decomposition (SVD)

## Definition

The singular value decomposition (SVD) is a factorization of any real or complex matrix into three components. It generalizes the eigendecomposition and provides a powerful tool in linear algebra, machine learning, and data science.

For an m×n real matrix A, the SVD can be expressed as:
**A = UΣV^T**

Where:
- **U** is an m×m orthogonal matrix (columns are left singular vectors)
- **Σ** is an m×n diagonal matrix with non-negative entries called singular values (σ₁ ≥ σ₂ ≥ ... ≥ σₙ ≥ 0)
- **V^T** is the transpose of an n×n orthogonal matrix (columns are right singular vectors)

## Geometric Interpretation

The SVD can be understood as three sequential operations:
1. A rotation/reflection (V^T) to align the coordinate axes
2. A scaling along the axes by the singular values (Σ)
3. Another rotation/reflection (U) to reach the final output

This means any matrix can be viewed as "rotate, scale, rotate".

## Key Properties

- The singular values are always non-negative real numbers
- The sum of squared singular values equals the Frobenius norm squared: Σσᵢ² = ||A||_F²
- Singular values represent the magnitudes of stretching in different directions
- SVD exists for any matrix and is unique up to the ordering of equal singular values

## Variations

There are three common types of SVD:
- **Full SVD**: U is m×m, Σ is m×n, V is n×n
- **Thin (Reduced) SVD**: Used when A is full rank
- **Compact SVD**: Often used in applications where only non-zero singular values matter

## Applications

SVD has wide-ranging applications including:
- Low-rank matrix approximation and data compression
- Image processing and noise reduction
- Principal Component Analysis (PCA)
- Solving linear systems and finding pseudoinverses
- Recommender systems
- Computer vision and robotics
- Signal processing

## Relation to Eigenvalue Decomposition

SVD is related to eigendecomposition for square matrices: if A is a normal matrix (AA^T = A^TA), then A = PDP^-1 where D contains the eigenvalues, SVD provides the same decomposition but works even when eigenvalues are not real or when eigenvectors are not orthogonal.

## Computing SVD

Common algorithms include:
- Jacobi algorithm (iterative method)
- Golub-Kahan bidiagonalization
- Divide and conquer methods
- Iterative refinement techniques

Computational complexity is typically O(min(m,n)n²) for full SVD.

---
*Last updated based on current information (2026)*