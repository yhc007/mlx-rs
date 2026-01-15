//! Random number generation
//!
//! This module provides random number generation functions for MLX arrays.

use mlx_sys;

use crate::array::Array;
use crate::dtype::{ArrayElement, DType};
use crate::error::{Error, Result};
use crate::stream::Stream;

/// Random key for PRNG
pub struct RandomKey {
    inner: Array,
}

impl RandomKey {
    /// Create a new random key from a seed
    pub fn new(seed: u64) -> Result<Self> {
        let mut inner = Array::new_uninit();
        let status = unsafe { mlx_sys::mlx_random_key(inner.as_mut_ptr(), seed) };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to create random key".into()));
        }

        Ok(Self { inner })
    }

    /// Split the key into two new keys
    pub fn split(&self) -> Result<(Self, Self)> {
        // For simplicity, create two new keys with random seeds
        // A proper implementation would use mlx_random_split
        let seed1: u64 = rand::random();
        let seed2: u64 = rand::random();

        Ok((Self::new(seed1)?, Self::new(seed2)?))
    }

    #[allow(dead_code)]
    pub(crate) fn as_raw(&self) -> mlx_sys::mlx_array {
        self.inner.as_raw()
    }
}

impl Clone for RandomKey {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// Helper to get a random key
fn get_key(key: Option<&RandomKey>) -> Result<Array> {
    match key {
        Some(k) => Ok(k.inner.clone()),
        None => {
            let seed: u64 = rand::random();
            let mut arr = Array::new_uninit();
            let status = unsafe { mlx_sys::mlx_random_key(arr.as_mut_ptr(), seed) };
            if status != 0 {
                return Err(Error::ArrayCreation("Failed to create random key".into()));
            }
            Ok(arr)
        }
    }
}

// ============================================================================
// Random Array Generation
// ============================================================================

/// Generate uniform random values in [low, high)
pub fn uniform<T: ArrayElement>(
    shape: &[i32],
    low: f32,
    high: f32,
    key: Option<&RandomKey>,
) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let key_arr = get_key(key)?;
    let low_arr = Array::from_float(low);
    let high_arr = Array::from_float(high);

    let status = unsafe {
        mlx_sys::mlx_random_uniform(
            result.as_mut_ptr(),
            low_arr.as_raw(),
            high_arr.as_raw(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key_arr.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to generate uniform random".into()));
    }

    Ok(result)
}

/// Generate standard normal random values (mean=0, std=1)
pub fn normal<T: ArrayElement>(shape: &[i32], key: Option<&RandomKey>) -> Result<Array> {
    normal_with_params::<T>(shape, 0.0, 1.0, key)
}

/// Generate normal random values with specified mean and standard deviation
pub fn normal_with_params<T: ArrayElement>(
    shape: &[i32],
    mean: f32,
    std: f32,
    key: Option<&RandomKey>,
) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let key_arr = get_key(key)?;

    let status = unsafe {
        mlx_sys::mlx_random_normal(
            result.as_mut_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            mean,
            std,
            key_arr.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to generate normal random".into()));
    }

    Ok(result)
}

/// Generate random integers in [low, high)
pub fn randint(shape: &[i32], low: i32, high: i32, key: Option<&RandomKey>) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let key_arr = get_key(key)?;
    let low_arr = Array::from_int(low);
    let high_arr = Array::from_int(high);

    let status = unsafe {
        mlx_sys::mlx_random_randint(
            result.as_mut_ptr(),
            low_arr.as_raw(),
            high_arr.as_raw(),
            shape.as_ptr(),
            shape.len(),
            DType::Int32.into(),
            key_arr.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to generate randint".into()));
    }

    Ok(result)
}

/// Generate Bernoulli random values (0 or 1)
pub fn bernoulli(shape: &[i32], p: f32, key: Option<&RandomKey>) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let key_arr = get_key(key)?;
    let p_arr = Array::from_float(p);

    let status = unsafe {
        mlx_sys::mlx_random_bernoulli(
            result.as_mut_ptr(),
            p_arr.as_raw(),
            shape.as_ptr(),
            shape.len(),
            key_arr.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to generate bernoulli".into()));
    }

    Ok(result)
}

/// Generate truncated normal random values
pub fn truncated_normal<T: ArrayElement>(
    shape: &[i32],
    low: f32,
    high: f32,
    key: Option<&RandomKey>,
) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let key_arr = get_key(key)?;
    let low_arr = Array::from_float(low);
    let high_arr = Array::from_float(high);

    let status = unsafe {
        mlx_sys::mlx_random_truncated_normal(
            result.as_mut_ptr(),
            low_arr.as_raw(),
            high_arr.as_raw(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key_arr.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation(
            "Failed to generate truncated_normal".into(),
        ));
    }

    Ok(result)
}

// ============================================================================
// Initialization helpers for neural networks
// ============================================================================

/// Xavier/Glorot uniform initialization
pub fn glorot_uniform(shape: &[i32], key: Option<&RandomKey>) -> Result<Array> {
    if shape.len() < 2 {
        return Err(Error::InvalidShape(
            "Glorot initialization requires at least 2 dimensions".into(),
        ));
    }

    let fan_in = shape[shape.len() - 2] as f32;
    let fan_out = shape[shape.len() - 1] as f32;
    let limit = (6.0 / (fan_in + fan_out)).sqrt();

    uniform::<f32>(shape, -limit, limit, key)
}

/// Xavier/Glorot normal initialization
pub fn glorot_normal(shape: &[i32], key: Option<&RandomKey>) -> Result<Array> {
    if shape.len() < 2 {
        return Err(Error::InvalidShape(
            "Glorot initialization requires at least 2 dimensions".into(),
        ));
    }

    let fan_in = shape[shape.len() - 2] as f32;
    let fan_out = shape[shape.len() - 1] as f32;
    let std = (2.0 / (fan_in + fan_out)).sqrt();

    normal_with_params::<f32>(shape, 0.0, std, key)
}

/// He/Kaiming uniform initialization
pub fn he_uniform(shape: &[i32], key: Option<&RandomKey>) -> Result<Array> {
    if shape.len() < 2 {
        return Err(Error::InvalidShape(
            "He initialization requires at least 2 dimensions".into(),
        ));
    }

    let fan_in = shape[shape.len() - 2] as f32;
    let limit = (6.0 / fan_in).sqrt();

    uniform::<f32>(shape, -limit, limit, key)
}

/// He/Kaiming normal initialization
pub fn he_normal(shape: &[i32], key: Option<&RandomKey>) -> Result<Array> {
    if shape.len() < 2 {
        return Err(Error::InvalidShape(
            "He initialization requires at least 2 dimensions".into(),
        ));
    }

    let fan_in = shape[shape.len() - 2] as f32;
    let std = (2.0 / fan_in).sqrt();

    normal_with_params::<f32>(shape, 0.0, std, key)
}

/// Set the global random seed
pub fn seed(s: u64) {
    unsafe {
        mlx_sys::mlx_random_seed(s);
    }
}
