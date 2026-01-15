//! Linear algebra operations
//!
//! This module provides linear algebra operations for MLX arrays,
//! including matrix decompositions, solvers, and norms.

use mlx_sys;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::stream::Stream;

// ============================================================================
// Matrix Inverse
// ============================================================================

/// Compute the inverse of a square matrix
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// Inverse of the matrix
pub fn inv(a: &Array) -> Result<Array> {
    let stream = Stream::cpu();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_inv(result.as_mut_ptr(), a.as_raw(), stream.as_raw())
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute matrix inverse".into()));
    }

    Ok(result)
}

/// Compute the Moore-Penrose pseudo-inverse
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Input matrix
///
/// # Returns
/// Pseudo-inverse of the matrix
pub fn pinv(a: &Array) -> Result<Array> {
    let stream = Stream::cpu();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_pinv(result.as_mut_ptr(), a.as_raw(), stream.as_raw())
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute pseudo-inverse".into()));
    }

    Ok(result)
}

/// Compute the inverse of a triangular matrix
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Input triangular matrix
/// * `upper` - If true, a is upper triangular; otherwise lower triangular
///
/// # Returns
/// Inverse of the triangular matrix
pub fn tri_inv(a: &Array, upper: bool) -> Result<Array> {
    let stream = Stream::cpu();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_tri_inv(result.as_mut_ptr(), a.as_raw(), upper, stream.as_raw())
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute triangular inverse".into()));
    }

    Ok(result)
}

// ============================================================================
// Linear System Solvers
// ============================================================================

/// Solve a linear system Ax = b
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Coefficient matrix A
/// * `b` - Right-hand side vector/matrix b
///
/// # Returns
/// Solution x to the system Ax = b
pub fn solve(a: &Array, b: &Array) -> Result<Array> {
    let stream = Stream::cpu();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_solve(
            result.as_mut_ptr(),
            a.as_raw(),
            b.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to solve linear system".into()));
    }

    Ok(result)
}

/// Solve a triangular linear system
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Triangular coefficient matrix
/// * `b` - Right-hand side vector/matrix
/// * `upper` - If true, a is upper triangular; otherwise lower triangular
///
/// # Returns
/// Solution to the triangular system
pub fn solve_triangular(a: &Array, b: &Array, upper: bool) -> Result<Array> {
    let stream = Stream::cpu();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_solve_triangular(
            result.as_mut_ptr(),
            a.as_raw(),
            b.as_raw(),
            upper,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to solve triangular system".into()));
    }

    Ok(result)
}

// ============================================================================
// Matrix Decompositions
// ============================================================================

/// Compute the Cholesky decomposition
///
/// Decomposes a positive definite matrix A into L * L^T (or U^T * U).
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Positive definite matrix
/// * `upper` - If true, compute upper triangular U; otherwise lower triangular L
///
/// # Returns
/// Cholesky factor (L or U)
pub fn cholesky(a: &Array, upper: bool) -> Result<Array> {
    // Cholesky requires CPU stream
    let stream = Stream::cpu();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_cholesky(result.as_mut_ptr(), a.as_raw(), upper, stream.as_raw())
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute Cholesky decomposition".into()));
    }

    Ok(result)
}

/// Compute the inverse using Cholesky decomposition
///
/// More efficient for positive definite matrices.
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Positive definite matrix
/// * `upper` - If true, treat a as upper triangular Cholesky factor
///
/// # Returns
/// Inverse of the matrix
pub fn cholesky_inv(a: &Array, upper: bool) -> Result<Array> {
    let stream = Stream::cpu();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_cholesky_inv(result.as_mut_ptr(), a.as_raw(), upper, stream.as_raw())
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute Cholesky inverse".into()));
    }

    Ok(result)
}

/// Compute the QR decomposition
///
/// Decomposes a matrix A into Q * R where Q is orthogonal and R is upper triangular.
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Input matrix
///
/// # Returns
/// Tuple (Q, R) where Q is orthogonal and R is upper triangular
pub fn qr(a: &Array) -> Result<(Array, Array)> {
    let stream = Stream::cpu();
    let mut q = Array::new_uninit();
    let mut r = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_qr(q.as_mut_ptr(), r.as_mut_ptr(), a.as_raw(), stream.as_raw())
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute QR decomposition".into()));
    }

    Ok((q, r))
}

/// Compute the Singular Value Decomposition (SVD)
///
/// Decomposes a matrix A into U * S * V^T.
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Input matrix
///
/// # Returns
/// Tuple (U, S, Vt) where U and Vt are orthogonal and S contains singular values
pub fn svd(a: &Array) -> Result<(Array, Array, Array)> {
    let stream = Stream::cpu();
    let mut result_vec = unsafe { mlx_sys::mlx_vector_array_new() };

    let status = unsafe {
        mlx_sys::mlx_linalg_svd(&mut result_vec, a.as_raw(), true, stream.as_raw())
    };

    if status != 0 {
        unsafe { mlx_sys::mlx_vector_array_free(result_vec) };
        return Err(Error::ArrayCreation("Failed to compute SVD".into()));
    }

    // Extract U, S, Vt from the vector
    let size = unsafe { mlx_sys::mlx_vector_array_size(result_vec) };
    if size != 3 {
        unsafe { mlx_sys::mlx_vector_array_free(result_vec) };
        return Err(Error::ArrayCreation("SVD returned unexpected number of arrays".into()));
    }

    let mut u = Array::new_uninit();
    let mut s = Array::new_uninit();
    let mut vt = Array::new_uninit();

    unsafe {
        mlx_sys::mlx_vector_array_get(u.as_mut_ptr(), result_vec, 0);
        mlx_sys::mlx_vector_array_get(s.as_mut_ptr(), result_vec, 1);
        mlx_sys::mlx_vector_array_get(vt.as_mut_ptr(), result_vec, 2);
        mlx_sys::mlx_vector_array_free(result_vec);
    }

    Ok((u, s, vt))
}

/// Compute singular values only (without U and V)
///
/// More efficient when only singular values are needed.
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Input matrix
///
/// # Returns
/// Array of singular values
pub fn svdvals(a: &Array) -> Result<Array> {
    let stream = Stream::cpu();
    let mut result_vec = unsafe { mlx_sys::mlx_vector_array_new() };

    let status = unsafe {
        mlx_sys::mlx_linalg_svd(&mut result_vec, a.as_raw(), false, stream.as_raw())
    };

    if status != 0 {
        unsafe { mlx_sys::mlx_vector_array_free(result_vec) };
        return Err(Error::ArrayCreation("Failed to compute singular values".into()));
    }

    let mut s = Array::new_uninit();
    unsafe {
        mlx_sys::mlx_vector_array_get(s.as_mut_ptr(), result_vec, 0);
        mlx_sys::mlx_vector_array_free(result_vec);
    }

    Ok(s)
}

/// Compute the LU decomposition
///
/// Decomposes a matrix A into P * L * U where P is a permutation matrix,
/// L is lower triangular, and U is upper triangular.
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Input matrix
///
/// # Returns
/// Tuple (P, L, U)
pub fn lu(a: &Array) -> Result<(Array, Array, Array)> {
    let stream = Stream::cpu();
    let mut result_vec = unsafe { mlx_sys::mlx_vector_array_new() };

    let status = unsafe {
        mlx_sys::mlx_linalg_lu(&mut result_vec, a.as_raw(), stream.as_raw())
    };

    if status != 0 {
        unsafe { mlx_sys::mlx_vector_array_free(result_vec) };
        return Err(Error::ArrayCreation("Failed to compute LU decomposition".into()));
    }

    let size = unsafe { mlx_sys::mlx_vector_array_size(result_vec) };
    if size != 3 {
        unsafe { mlx_sys::mlx_vector_array_free(result_vec) };
        return Err(Error::ArrayCreation("LU returned unexpected number of arrays".into()));
    }

    let mut p = Array::new_uninit();
    let mut l = Array::new_uninit();
    let mut u = Array::new_uninit();

    unsafe {
        mlx_sys::mlx_vector_array_get(p.as_mut_ptr(), result_vec, 0);
        mlx_sys::mlx_vector_array_get(l.as_mut_ptr(), result_vec, 1);
        mlx_sys::mlx_vector_array_get(u.as_mut_ptr(), result_vec, 2);
        mlx_sys::mlx_vector_array_free(result_vec);
    }

    Ok((p, l, u))
}

// ============================================================================
// Eigenvalue Decomposition
// ============================================================================

/// Compute eigenvalues and eigenvectors
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// Tuple (eigenvalues, eigenvectors)
pub fn eig(a: &Array) -> Result<(Array, Array)> {
    let stream = Stream::cpu();
    let mut eigenvalues = Array::new_uninit();
    let mut eigenvectors = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_eig(
            eigenvalues.as_mut_ptr(),
            eigenvectors.as_mut_ptr(),
            a.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute eigendecomposition".into()));
    }

    Ok((eigenvalues, eigenvectors))
}

/// Compute eigenvalues only
///
/// More efficient when eigenvectors are not needed.
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// Array of eigenvalues
pub fn eigvals(a: &Array) -> Result<Array> {
    let stream = Stream::cpu();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_eigvals(result.as_mut_ptr(), a.as_raw(), stream.as_raw())
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute eigenvalues".into()));
    }

    Ok(result)
}

/// Compute eigenvalues and eigenvectors of a Hermitian matrix
///
/// More efficient for symmetric/Hermitian matrices.
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Hermitian/symmetric matrix
///
/// # Returns
/// Tuple (eigenvalues, eigenvectors)
pub fn eigh(a: &Array) -> Result<(Array, Array)> {
    let stream = Stream::cpu();
    let mut eigenvalues = Array::new_uninit();
    let mut eigenvectors = Array::new_uninit();

    let uplo = std::ffi::CString::new("L").unwrap();

    let status = unsafe {
        mlx_sys::mlx_linalg_eigh(
            eigenvalues.as_mut_ptr(),
            eigenvectors.as_mut_ptr(),
            a.as_raw(),
            uplo.as_ptr(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute Hermitian eigendecomposition".into()));
    }

    Ok((eigenvalues, eigenvectors))
}

/// Compute eigenvalues of a Hermitian matrix
///
/// Note: This operation currently requires CPU execution.
///
/// # Arguments
/// * `a` - Hermitian/symmetric matrix
///
/// # Returns
/// Array of eigenvalues (real)
pub fn eigvalsh(a: &Array) -> Result<Array> {
    let stream = Stream::cpu();
    let mut result = Array::new_uninit();

    let uplo = std::ffi::CString::new("L").unwrap();

    let status = unsafe {
        mlx_sys::mlx_linalg_eigvalsh(result.as_mut_ptr(), a.as_raw(), uplo.as_ptr(), stream.as_raw())
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute Hermitian eigenvalues".into()));
    }

    Ok(result)
}

// ============================================================================
// Norms
// ============================================================================

/// Compute the L2 norm (Euclidean norm) of the entire array
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// L2 norm of the array
pub fn norm_l2(a: &Array) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_norm_l2(
            result.as_mut_ptr(),
            a.as_raw(),
            std::ptr::null(),
            0,
            false,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute L2 norm".into()));
    }

    Ok(result)
}

/// Compute the L2 norm along specified axes
///
/// # Arguments
/// * `a` - Input array
/// * `axis` - Axes along which to compute the norm
/// * `keepdims` - If true, keep the reduced dimensions with size 1
///
/// # Returns
/// L2 norm along the specified axes
pub fn norm_l2_axis(a: &Array, axis: &[i32], keepdims: bool) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_norm_l2(
            result.as_mut_ptr(),
            a.as_raw(),
            axis.as_ptr(),
            axis.len(),
            keepdims,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute L2 norm".into()));
    }

    Ok(result)
}

/// Compute the vector norm
///
/// # Arguments
/// * `a` - Input array
/// * `ord` - Order of the norm (e.g., 1.0 for L1, 2.0 for L2, f64::INFINITY for max)
/// * `axis` - Axis along which to compute the norm
/// * `keepdims` - If true, keep the reduced dimension with size 1
///
/// # Returns
/// Norm of the array
pub fn norm(a: &Array, ord: f64, axis: Option<&[i32]>, keepdims: bool) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let (axis_ptr, axis_len) = match axis {
        Some(ax) => (ax.as_ptr(), ax.len()),
        None => (std::ptr::null(), 0),
    };

    let status = unsafe {
        mlx_sys::mlx_linalg_norm(
            result.as_mut_ptr(),
            a.as_raw(),
            ord,
            axis_ptr,
            axis_len,
            keepdims,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute norm".into()));
    }

    Ok(result)
}

// ============================================================================
// Other Operations
// ============================================================================

/// Compute the cross product of two 3D vectors
///
/// # Arguments
/// * `a` - First vector (must have size 3 along the specified axis)
/// * `b` - Second vector (must have size 3 along the specified axis)
/// * `axis` - Axis along which to compute the cross product
///
/// # Returns
/// Cross product a × b
pub fn cross(a: &Array, b: &Array, axis: i32) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_linalg_cross(
            result.as_mut_ptr(),
            a.as_raw(),
            b.as_raw(),
            axis,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute cross product".into()));
    }

    Ok(result)
}
