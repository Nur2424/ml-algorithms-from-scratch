# PCA & SVD From Scratch
 
A step-by-step, intuition-first implementation of Principal Component Analysis and Singular Value Decomposition — from theory to working NumPy code.
 
---
 
## What This Is
 
Most PCA tutorials hand you `sklearn.decomposition.PCA` and call it a day.
 
This repo does the opposite.
 
Every concept is built from the ground up — starting from *why variance matters*, moving through the geometry, deriving the math, and ending with a full scratch implementation. No black boxes.
 
---
 
## Files
 
### `theory.md`
 
A structured walkthrough of everything behind PCA and SVD:
 
- **Intuition** — why high-dimensional data is a problem and what dimensionality reduction actually means
- **Geometry** — variance, projections, and what "best direction" means visually
- **Linear algebra** — vectors, dot products, norms, orthogonality — introduced only when needed
- **PCA derivation** — covariance matrix → eigenvalues → eigenvectors → projection
- **SVD** — the rotate → scale → rotate decomposition, explained geometrically
- **PCA ↔ SVD connection** — why they give the same result and why SVD is preferred in practice
 
> Focus: deep understanding, not formula memorization.
 
---
 
### `code-implement-Copy1.ipynb`
 
A Jupyter Notebook implementing PCA from scratch using only NumPy:
 
- Data centering
- Covariance matrix computation
- Eigen decomposition
- Sorting principal components by variance
- Projection to lower dimensions
- Variance explained per component
- Reconstruction and reconstruction error
 
> No sklearn. Everything is written manually to show what is actually happening at each step.
 
---
 
## Key Takeaways
 
- PCA finds the directions of maximum variance in your data
- Projection is just a dot product
- Eigenvectors are the principal directions
- SVD decomposes any matrix into rotate → scale → rotate
- PCA and SVD on centered data give identical results
- sklearn's `PCA` uses SVD internally — now you know why
 
---
 
## Who This Is For
 
Anyone who wants to move from *"I've heard of PCA"* to *"I can derive it, implement it, and explain it clearly"*.
 
---
 
> If you run the notebook without reading the theory, you will miss the point entirely.
> This repo is built for understanding — not just results.
 
