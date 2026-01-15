//! # MLX-RS - Apple's Machine Learning Framework for Rust
//!
//! Safe, idiomatic Rust bindings for Apple's MLX array framework.
//!
//! MLX is an array framework designed for efficient machine learning research
//! on Apple silicon. This crate provides a safe, Rust-native API while
//! leveraging MLX's lazy evaluation and unified memory architecture.
//!
//! ## Features
//!
//! - **Safe API**: All unsafe FFI calls are wrapped in safe Rust abstractions
//! - **Lazy Evaluation**: Operations are only computed when needed
//! - **Unified Memory**: Seamless CPU/GPU computation on Apple silicon
//! - **Rust Idioms**: Implements standard traits like `Add`, `Mul`, `Drop`, etc.
//!
//! ## Example
//!
//! ```ignore
//! use mlx_rs::prelude::*;
//!
//! // Create arrays
//! let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
//! let b = Array::from_slice(&[4.0f32, 5.0, 6.0], &[3])?;
//!
//! // Arithmetic operations (lazy)
//! let c = &a + &b;
//! let d = &a * &b;
//!
//! // Evaluate and get results
//! c.eval();
//! println!("Sum: {:?}", c.to_vec::<f32>()?);
//! ```
//!
//! ## Architecture
//!
//! The crate is structured in two layers:
//!
//! - `mlx-sys`: Low-level, unsafe FFI bindings to mlx-c
//! - `mlx-rs`: High-level, safe Rust API (this crate)

pub mod array;
pub mod device;
pub mod dtype;
pub mod error;
pub mod linalg;
pub mod nn;
pub mod ops;
pub mod random;
pub mod stream;
pub mod transforms;

pub use array::Array;
pub use device::{Device, DeviceType};
pub use dtype::DType;
pub use error::{Error, Result};
pub use stream::Stream;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::array::Array;
    pub use crate::device::{Device, DeviceType};
    pub use crate::dtype::DType;
    pub use crate::error::{Error, Result};
    pub use crate::linalg;
    pub use crate::nn;
    pub use crate::ops::*;
    pub use crate::random;
    pub use crate::stream::Stream;
}

/// Initialize MLX with default settings
///
/// This should be called once at the start of your program.
/// It sets up the default device and stream.
pub fn init() {
    // Initialize with default device (GPU if available, else CPU)
    let device = Device::default_device();
    Device::set_default(&device);
}

/// Synchronize all pending operations on the default stream
pub fn synchronize() {
    let device = Device::default_device();
    let stream = Stream::default_stream(&device);
    stream.synchronize();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation() {
        let arr = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(arr.shape(), vec![2, 2]);
        assert_eq!(arr.size(), 4);
        assert_eq!(arr.ndim(), 2);
    }

    #[test]
    fn test_array_zeros_ones() {
        let zeros = Array::zeros::<f32>(&[3, 3]).unwrap();
        zeros.eval();
        let data = zeros.to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));

        let ones = Array::ones::<f32>(&[2, 2]).unwrap();
        ones.eval();
        let data = ones.to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_array_arithmetic() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = Array::from_slice(&[4.0f32, 5.0, 6.0], &[3]).unwrap();

        let sum = &a + &b;
        sum.eval();
        assert_eq!(sum.to_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);

        let diff = &a - &b;
        diff.eval();
        assert_eq!(diff.to_vec::<f32>().unwrap(), vec![-3.0, -3.0, -3.0]);

        let prod = &a * &b;
        prod.eval();
        assert_eq!(prod.to_vec::<f32>().unwrap(), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_array_reshape() {
        let arr = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let reshaped = arr.reshape(&[3, 2]).unwrap();
        reshaped.eval();
        assert_eq!(reshaped.shape(), vec![3, 2]);
    }

    #[test]
    fn test_array_transpose() {
        let arr = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let transposed = arr.transpose().unwrap();
        transposed.eval();
        assert_eq!(transposed.shape(), vec![3, 2]);
    }

    #[test]
    fn test_array_reduction() {
        let arr = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

        let sum = arr.sum_all(false).unwrap();
        sum.eval();
        assert_eq!(sum.to_vec::<f32>().unwrap(), vec![10.0]);

        let mean = arr.mean_all(false).unwrap();
        mean.eval();
        assert_eq!(mean.to_vec::<f32>().unwrap(), vec![2.5]);
    }

    #[test]
    fn test_matmul() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let c = a.matmul(&b).unwrap();
        c.eval();
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_nn_relu() {
        let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let y = nn::relu(&x).unwrap();
        y.eval();
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_nn_sigmoid() {
        let x = Array::from_float(0.0);
        let y = nn::sigmoid(&x).unwrap();
        y.eval();
        let val = y.to_vec::<f32>().unwrap()[0];
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_random_uniform() {
        let arr = random::uniform::<f32>(&[100], 0.0, 1.0, None).unwrap();
        arr.eval();
        let data = arr.to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_concatenate() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let b = Array::from_slice(&[3.0f32, 4.0], &[2]).unwrap();

        let c = ops::concatenate(&[&a, &b], 0).unwrap();
        c.eval();
        assert_eq!(c.shape(), vec![4]);
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_stack() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let b = Array::from_slice(&[3.0f32, 4.0], &[2]).unwrap();

        let c = ops::stack(&[&a, &b], 0).unwrap();
        c.eval();
        assert_eq!(c.shape(), vec![2, 2]);
    }

    #[test]
    fn test_slice() {
        let arr = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let sliced = arr.slice(&[0, 0], &[1, 3], None).unwrap();
        sliced.eval();
        assert_eq!(sliced.shape(), vec![1, 3]);
        assert_eq!(sliced.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_comparison_eq_ne() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = Array::from_slice(&[1.0f32, 5.0, 3.0, 6.0], &[4]).unwrap();

        let eq_result = a.eq(&b).unwrap();
        eq_result.eval();
        // Results should be [true, false, true, false]

        let ne_result = a.ne(&b).unwrap();
        ne_result.eval();
        // Results should be [false, true, false, true]
    }

    #[test]
    fn test_comparison_lt_gt() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = Array::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4]).unwrap();

        let lt_result = a.lt(&b).unwrap();
        lt_result.eval();
        // [1<2, 2<2, 3<2, 4<2] = [true, false, false, false]

        let gt_result = a.gt(&b).unwrap();
        gt_result.eval();
        // [1>2, 2>2, 3>2, 4>2] = [false, false, true, true]

        let le_result = a.le(&b).unwrap();
        le_result.eval();
        // [1<=2, 2<=2, 3<=2, 4<=2] = [true, true, false, false]

        let ge_result = a.ge(&b).unwrap();
        ge_result.eval();
        // [1>=2, 2>=2, 3>=2, 4>=2] = [false, true, true, true]
    }

    #[test]
    fn test_logical_ops() {
        let a = Array::from_slice(&[true, true, false, false], &[4]).unwrap();
        let b = Array::from_slice(&[true, false, true, false], &[4]).unwrap();

        let and_result = a.logical_and(&b).unwrap();
        and_result.eval();
        // [T&T, T&F, F&T, F&F] = [true, false, false, false]

        let or_result = a.logical_or(&b).unwrap();
        or_result.eval();
        // [T|T, T|F, F|T, F|F] = [true, true, true, false]

        let not_result = a.logical_not().unwrap();
        not_result.eval();
        // [!T, !T, !F, !F] = [false, false, true, true]
    }

    #[test]
    fn test_where_cond() {
        let cond = Array::from_slice(&[true, false, true, false], &[4]).unwrap();
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let y = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4]).unwrap();

        let result = ops::where_cond(&cond, &x, &y).unwrap();
        result.eval();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn test_conv1d() {
        // MLX uses channels-last format: (N, L, C_in)
        // Input: batch=1, length=5, channels=1
        let input = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 5, 1]).unwrap();
        // Kernel: out_channels=1, kernel_size=3, in_channels=1
        let kernel = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1]).unwrap();

        let output = nn::conv1d(&input, &kernel, 1, 0, 1, 1).unwrap();
        output.eval();

        // Output shape should be (1, 3, 1) - (5 - 3 + 1 = 3)
        assert_eq!(output.shape(), vec![1, 3, 1]);

        // Values: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
        let values = output.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_conv1d_with_padding() {
        // MLX uses channels-last format: (N, L, C_in)
        let input = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 5, 1]).unwrap();
        let kernel = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1]).unwrap();

        // With padding=1, output length = 5 - 3 + 2*1 + 1 = 5
        let output = nn::conv1d(&input, &kernel, 1, 1, 1, 1).unwrap();
        output.eval();

        assert_eq!(output.shape(), vec![1, 5, 1]);
    }

    #[test]
    fn test_conv2d() {
        // MLX uses channels-last format: (N, H, W, C_in)
        // Input: batch=1, height=4, width=4, channels=1
        let input = Array::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            &[1, 4, 4, 1],
        ).unwrap();

        // Kernel: out_channels=1, kH=2, kW=2, in_channels=1
        let kernel = Array::from_slice(
            &[1.0f32, 1.0, 1.0, 1.0],
            &[1, 2, 2, 1],
        ).unwrap();

        let output = nn::conv2d(&input, &kernel, (1, 1), (0, 0), (1, 1), 1).unwrap();
        output.eval();

        // Output shape: (1, 3, 3, 1) - (4-2+1 = 3)
        assert_eq!(output.shape(), vec![1, 3, 3, 1]);

        // First value: 1+2+5+6 = 14
        let values = output.to_vec::<f32>().unwrap();
        assert_eq!(values[0], 14.0);
    }

    #[test]
    fn test_conv2d_simple() {
        // MLX uses channels-last format: (N, H, W, C_in)
        let input = Array::from_slice(
            &[1.0f32; 16],
            &[1, 4, 4, 1],
        ).unwrap();

        // Kernel: out_channels=1, kH=3, kW=3, in_channels=1
        let kernel = Array::from_slice(
            &[1.0f32; 9],
            &[1, 3, 3, 1],
        ).unwrap();

        let output = nn::conv2d_simple(&input, &kernel).unwrap();
        output.eval();

        // Output shape: (1, 2, 2, 1)
        assert_eq!(output.shape(), vec![1, 2, 2, 1]);

        // Each output is sum of 9 ones = 9
        let values = output.to_vec::<f32>().unwrap();
        assert!(values.iter().all(|&v| v == 9.0));
    }

    #[test]
    fn test_max_pool1d() {
        // MLX uses channels-last format: (N, L, C)
        // Input: batch=1, length=6, channels=1
        let input = Array::from_slice(&[1.0f32, 3.0, 2.0, 4.0, 6.0, 5.0], &[1, 6, 1]).unwrap();

        // Max pool with kernel_size=2, stride=2 (default)
        let output = nn::max_pool1d(&input, 2, None).unwrap();
        output.eval();

        // Output shape: (1, 3, 1) - (6 / 2 = 3)
        assert_eq!(output.shape(), vec![1, 3, 1]);

        // Values: max(1,3)=3, max(2,4)=4, max(6,5)=6
        let values = output.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_max_pool1d_stride1() {
        // MLX uses channels-last format: (N, L, C)
        let input = Array::from_slice(&[1.0f32, 3.0, 2.0, 4.0, 6.0], &[1, 5, 1]).unwrap();

        // Max pool with kernel_size=3, stride=1
        let output = nn::max_pool1d(&input, 3, Some(1)).unwrap();
        output.eval();

        // Output shape: (1, 3, 1) - (5 - 3 + 1 = 3)
        assert_eq!(output.shape(), vec![1, 3, 1]);

        // Values: max(1,3,2)=3, max(3,2,4)=4, max(2,4,6)=6
        let values = output.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_avg_pool1d() {
        // MLX uses channels-last format: (N, L, C)
        let input = Array::from_slice(&[1.0f32, 3.0, 2.0, 4.0, 6.0, 8.0], &[1, 6, 1]).unwrap();

        // Avg pool with kernel_size=2, stride=2 (default)
        let output = nn::avg_pool1d(&input, 2, None).unwrap();
        output.eval();

        // Output shape: (1, 3, 1)
        assert_eq!(output.shape(), vec![1, 3, 1]);

        // Values: avg(1,3)=2, avg(2,4)=3, avg(6,8)=7
        let values = output.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![2.0, 3.0, 7.0]);
    }

    #[test]
    fn test_max_pool2d() {
        // MLX uses channels-last format: (N, H, W, C)
        // Input: batch=1, height=4, width=4, channels=1
        let input = Array::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            &[1, 4, 4, 1],
        ).unwrap();

        // Max pool with kernel_size=(2,2), stride=(2,2)
        let output = nn::max_pool2d(&input, (2, 2), None).unwrap();
        output.eval();

        // Output shape: (1, 2, 2, 1)
        assert_eq!(output.shape(), vec![1, 2, 2, 1]);

        // Values: max of each 2x2 block
        // Block 1: max(1,2,5,6) = 6
        // Block 2: max(3,4,7,8) = 8
        // Block 3: max(9,10,13,14) = 14
        // Block 4: max(11,12,15,16) = 16
        let values = output.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avg_pool2d() {
        // MLX uses channels-last format: (N, H, W, C)
        let input = Array::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            &[1, 4, 4, 1],
        ).unwrap();

        // Avg pool with kernel_size=(2,2), stride=(2,2)
        let output = nn::avg_pool2d(&input, (2, 2), None).unwrap();
        output.eval();

        // Output shape: (1, 2, 2, 1)
        assert_eq!(output.shape(), vec![1, 2, 2, 1]);

        // Values: avg of each 2x2 block
        // Block 1: avg(1,2,5,6) = 3.5
        // Block 2: avg(3,4,7,8) = 5.5
        // Block 3: avg(9,10,13,14) = 11.5
        // Block 4: avg(11,12,15,16) = 13.5
        let values = output.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_argmax() {
        let arr = Array::from_slice(&[1.0f32, 5.0, 3.0, 2.0, 4.0], &[5]).unwrap();

        let idx = arr.argmax(false).unwrap();
        idx.eval();

        // Maximum is 5.0 at index 1
        let values = idx.to_vec::<u32>().unwrap();
        assert_eq!(values, vec![1]);
    }

    #[test]
    fn test_argmax_axis() {
        let arr = Array::from_slice(
            &[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0],
            &[2, 3],
        ).unwrap();

        // Find argmax along axis 1 (columns)
        let idx = arr.argmax_axis(1, false).unwrap();
        idx.eval();

        // Row 0: [1, 5, 3] -> max at index 1
        // Row 1: [4, 2, 6] -> max at index 2
        let values = idx.to_vec::<u32>().unwrap();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn test_argmin() {
        let arr = Array::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5]).unwrap();

        let idx = arr.argmin(false).unwrap();
        idx.eval();

        // Minimum is 1.0, first occurrence at index 1
        let values = idx.to_vec::<u32>().unwrap();
        assert_eq!(values, vec![1]);
    }

    #[test]
    fn test_argmin_axis() {
        let arr = Array::from_slice(
            &[3.0f32, 1.0, 4.0, 2.0, 5.0, 0.0],
            &[2, 3],
        ).unwrap();

        // Find argmin along axis 1 (columns)
        let idx = arr.argmin_axis(1, false).unwrap();
        idx.eval();

        // Row 0: [3, 1, 4] -> min at index 1
        // Row 1: [2, 5, 0] -> min at index 2
        let values = idx.to_vec::<u32>().unwrap();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn test_argsort() {
        let arr = Array::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5]).unwrap();

        let indices = arr.argsort().unwrap();
        indices.eval();

        // Sorted order: 1.0(1), 1.0(3), 3.0(0), 4.0(2), 5.0(4)
        let values = indices.to_vec::<u32>().unwrap();
        // Indices that sort the array
        assert_eq!(values.len(), 5);
        // First two should be indices 1 and 3 (both have value 1.0)
        assert!(values[0] == 1 || values[0] == 3);
    }

    #[test]
    fn test_sort() {
        let arr = Array::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0], &[5]).unwrap();

        let sorted = arr.sort().unwrap();
        sorted.eval();

        let values = sorted.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sort_axis() {
        let arr = Array::from_slice(
            &[3.0f32, 1.0, 4.0, 2.0, 5.0, 0.0],
            &[2, 3],
        ).unwrap();

        // Sort along axis 1 (each row)
        let sorted = arr.sort_axis(1).unwrap();
        sorted.eval();

        // Row 0: [3, 1, 4] -> [1, 3, 4]
        // Row 1: [2, 5, 0] -> [0, 2, 5]
        let values = sorted.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![1.0, 3.0, 4.0, 0.0, 2.0, 5.0]);
    }

    #[test]
    fn test_topk() {
        let arr = Array::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], &[7]).unwrap();

        let top3 = arr.topk(3).unwrap();
        top3.eval();

        // Top 3 values are 9, 5, 4 (not necessarily in order)
        let values = top3.to_vec::<f32>().unwrap();
        assert_eq!(values.len(), 3);

        // Sort and check
        let mut sorted_vals = values.clone();
        sorted_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(sorted_vals, vec![9.0, 5.0, 4.0]);
    }

    #[test]
    fn test_topk_axis() {
        let arr = Array::from_slice(
            &[3.0f32, 1.0, 4.0, 5.0, 2.0, 6.0, 0.0],
            &[7],
        ).unwrap();

        let top2 = arr.topk(2).unwrap();
        top2.eval();

        let values = top2.to_vec::<f32>().unwrap();
        assert_eq!(values.len(), 2);

        // Should contain the two largest: 6 and 5
        let mut sorted_vals = values.clone();
        sorted_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(sorted_vals, vec![6.0, 5.0]);
    }

    #[test]
    fn test_linalg_inv() {
        // 2x2 matrix: [[4, 7], [2, 6]]
        // Inverse: [[0.6, -0.7], [-0.2, 0.4]]
        let a = Array::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2]).unwrap();

        let a_inv = linalg::inv(&a).unwrap();
        a_inv.eval();

        let values = a_inv.to_vec::<f32>().unwrap();
        assert!((values[0] - 0.6).abs() < 1e-5);
        assert!((values[1] - (-0.7)).abs() < 1e-5);
        assert!((values[2] - (-0.2)).abs() < 1e-5);
        assert!((values[3] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_linalg_inv_identity() {
        // Inverse of identity is identity
        let eye = Array::eye::<f32>(3, None, 0).unwrap();

        let eye_inv = linalg::inv(&eye).unwrap();
        eye_inv.eval();

        let values = eye_inv.to_vec::<f32>().unwrap();
        // Check diagonal elements are 1
        assert!((values[0] - 1.0).abs() < 1e-5);
        assert!((values[4] - 1.0).abs() < 1e-5);
        assert!((values[8] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linalg_solve() {
        // Solve Ax = b where A = [[3, 1], [1, 2]], b = [9, 8]
        // Solution: x = [2, 3]
        let a = Array::from_slice(&[3.0f32, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        let b = Array::from_slice(&[9.0f32, 8.0], &[2]).unwrap();

        let x = linalg::solve(&a, &b).unwrap();
        x.eval();

        let values = x.to_vec::<f32>().unwrap();
        assert!((values[0] - 2.0).abs() < 1e-4);
        assert!((values[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_linalg_qr() {
        // 2x2 matrix
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let (q, r) = linalg::qr(&a).unwrap();
        q.eval();
        r.eval();

        // Q should be orthogonal (Q^T * Q = I)
        // R should be upper triangular (r[1][0] should be ~0)
        let r_vals = r.to_vec::<f32>().unwrap();

        // Check R is upper triangular
        assert!(r_vals[2].abs() < 1e-5); // r[1][0] should be 0
    }

    #[test]
    fn test_linalg_cholesky() {
        // Positive definite matrix: [[4, 2], [2, 2]]
        let a = Array::from_slice(&[4.0f32, 2.0, 2.0, 2.0], &[2, 2]).unwrap();

        let l = linalg::cholesky(&a, false).unwrap();
        l.eval();

        // L should be lower triangular
        let values = l.to_vec::<f32>().unwrap();
        assert!(values[1].abs() < 1e-5); // l[0][1] should be 0
    }

    #[test]
    fn test_linalg_svd() {
        // Simple 2x2 matrix
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let (u, s, vt) = linalg::svd(&a).unwrap();
        u.eval();
        s.eval();
        vt.eval();

        // S should have 2 singular values
        assert_eq!(s.shape(), vec![2]);

        // Singular values should be positive and sorted in descending order
        let s_vals = s.to_vec::<f32>().unwrap();
        assert!(s_vals[0] > 0.0);
        assert!(s_vals[1] > 0.0);
        assert!(s_vals[0] >= s_vals[1]);
    }

    #[test]
    fn test_linalg_eigvalsh() {
        // Symmetric matrix: [[2, 1], [1, 2]]
        // Eigenvalues: 3 and 1
        let a = Array::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2]).unwrap();

        let eigenvalues = linalg::eigvalsh(&a).unwrap();
        eigenvalues.eval();

        let values = eigenvalues.to_vec::<f32>().unwrap();
        assert_eq!(values.len(), 2);

        // Sort eigenvalues for comparison
        let mut sorted_vals = values.clone();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted_vals[0] - 1.0).abs() < 1e-4);
        assert!((sorted_vals[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_linalg_norm_l2() {
        // Vector [3, 4], L2 norm = 5
        let a = Array::from_slice(&[3.0f32, 4.0], &[2]).unwrap();

        let norm = linalg::norm_l2(&a).unwrap();
        norm.eval();

        let values = norm.to_vec::<f32>().unwrap();
        assert!((values[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_linalg_norm() {
        // Vector [1, 2, 3], L1 norm = 6, L2 norm = sqrt(14) ≈ 3.742
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();

        // L1 norm
        let l1_norm = linalg::norm(&a, 1.0, None, false).unwrap();
        l1_norm.eval();
        let l1_val = l1_norm.to_vec::<f32>().unwrap()[0];
        assert!((l1_val - 6.0).abs() < 1e-5);

        // L2 norm
        let l2_norm = linalg::norm(&a, 2.0, None, false).unwrap();
        l2_norm.eval();
        let l2_val = l2_norm.to_vec::<f32>().unwrap()[0];
        assert!((l2_val - 14.0_f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_linalg_cross() {
        // Cross product of [1, 0, 0] and [0, 1, 0] = [0, 0, 1]
        let a = Array::from_slice(&[1.0f32, 0.0, 0.0], &[3]).unwrap();
        let b = Array::from_slice(&[0.0f32, 1.0, 0.0], &[3]).unwrap();

        let cross = linalg::cross(&a, &b, 0).unwrap();
        cross.eval();

        let values = cross.to_vec::<f32>().unwrap();
        assert!((values[0] - 0.0).abs() < 1e-5);
        assert!((values[1] - 0.0).abs() < 1e-5);
        assert!((values[2] - 1.0).abs() < 1e-5);
    }

    // ============================================================================
    // Normalization Tests
    // ============================================================================

    #[test]
    fn test_layer_norm() {
        // Input: (batch, features) = (2, 4)
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        let weight = Array::ones::<f32>(&[4]).unwrap();
        let bias = Array::zeros::<f32>(&[4]).unwrap();

        let result = nn::layer_norm(&x, &weight, &bias, 1e-5).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![2, 4]);

        // Check that the normalized values have mean ~0 and std ~1 per row
        let values = result.to_vec::<f32>().unwrap();
        // First row: normalized [1, 2, 3, 4]
        let row1_mean: f32 = values[0..4].iter().sum::<f32>() / 4.0;
        assert!(row1_mean.abs() < 1e-5, "Row 1 mean should be ~0, got {}", row1_mean);
    }

    #[test]
    fn test_layer_norm_no_params() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();

        let result = nn::layer_norm_no_params(&x, 1e-5).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![1, 4]);

        let values = result.to_vec::<f32>().unwrap();
        let mean: f32 = values.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_rms_norm() {
        // Input: (batch, features) = (2, 4)
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        let weight = Array::ones::<f32>(&[4]).unwrap();

        let result = nn::rms_norm(&x, &weight, 1e-5).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![2, 4]);
    }

    #[test]
    fn test_rms_norm_no_params() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();

        let result = nn::rms_norm_no_params(&x, 1e-5).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![1, 4]);
    }

    #[test]
    fn test_batch_norm() {
        // Input: (batch, height, width, channels) = (2, 2, 2, 3)
        // For simplicity, use (N, C) = (4, 3) which is 2D batch norm
        let x = Array::from_slice(
            &[
                1.0f32, 2.0, 3.0,  // sample 1
                4.0, 5.0, 6.0,    // sample 2
                7.0, 8.0, 9.0,    // sample 3
                10.0, 11.0, 12.0, // sample 4
            ],
            &[4, 3],
        )
        .unwrap();

        let weight = Array::ones::<f32>(&[3]).unwrap();
        let bias = Array::zeros::<f32>(&[3]).unwrap();

        let result = nn::batch_norm(&x, &weight, &bias, None, None, true, 1e-5).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![4, 3]);

        // After batch norm, each channel should have mean ~0
        let values = result.to_vec::<f32>().unwrap();
        // Check first channel (indices 0, 3, 6, 9)
        let ch0_mean = (values[0] + values[3] + values[6] + values[9]) / 4.0;
        assert!(ch0_mean.abs() < 1e-4, "Channel 0 mean should be ~0, got {}", ch0_mean);
    }

    #[test]
    fn test_instance_norm() {
        // Input: (batch, length, channels) = (2, 4, 3)
        let x = Array::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,  // batch 1
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, // batch 2
            ],
            &[2, 4, 3],
        )
        .unwrap();

        let weight = Array::ones::<f32>(&[3]).unwrap();
        let bias = Array::zeros::<f32>(&[3]).unwrap();

        let result = nn::instance_norm(&x, &weight, &bias, 1e-5).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![2, 4, 3]);
    }

    #[test]
    fn test_group_norm() {
        // Input: (batch, channels) = (2, 4) with 2 groups
        let x = Array::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        )
        .unwrap();

        let weight = Array::ones::<f32>(&[4]).unwrap();
        let bias = Array::zeros::<f32>(&[4]).unwrap();

        let result = nn::group_norm(&x, 2, &weight, &bias, 1e-5).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![2, 4]);
    }

    #[test]
    fn test_group_norm_invalid_groups() {
        // 4 channels cannot be divided into 3 groups
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let weight = Array::ones::<f32>(&[4]).unwrap();
        let bias = Array::zeros::<f32>(&[4]).unwrap();

        let result = nn::group_norm(&x, 3, &weight, &bias, 1e-5);
        assert!(result.is_err());
    }

    // ============================================================================
    // Dropout Tests
    // ============================================================================

    #[test]
    fn test_dropout_training() {
        // Set seed for reproducibility
        random::seed(42);

        let x = Array::ones::<f32>(&[100, 100]).unwrap();

        // Apply dropout with p=0.5
        let result = nn::dropout(&x, 0.5, true).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![100, 100]);

        // Check that some elements are zeroed (with high probability)
        let values = result.to_vec::<f32>().unwrap();
        let zeros = values.iter().filter(|&&v| v == 0.0).count();
        let non_zeros = values.iter().filter(|&&v| v != 0.0).count();

        // With p=0.5, roughly half should be zero
        assert!(zeros > 2000, "Expected many zeros, got {}", zeros);
        assert!(non_zeros > 2000, "Expected many non-zeros, got {}", non_zeros);

        // Non-zero elements should be scaled by 1/(1-0.5) = 2.0
        let non_zero_val = values.iter().find(|&&v| v != 0.0).unwrap();
        assert!((*non_zero_val - 2.0).abs() < 1e-5, "Expected scaled value ~2.0, got {}", non_zero_val);
    }

    #[test]
    fn test_dropout_eval_mode() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // In eval mode, dropout should return input unchanged
        let result = nn::dropout(&x, 0.5, false).unwrap();
        result.eval();

        let x_values = x.to_vec::<f32>().unwrap();
        let result_values = result.to_vec::<f32>().unwrap();

        for (a, b) in x_values.iter().zip(result_values.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dropout_zero_probability() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // With p=0, all elements should be kept
        let result = nn::dropout(&x, 0.0, true).unwrap();
        result.eval();

        let x_values = x.to_vec::<f32>().unwrap();
        let result_values = result.to_vec::<f32>().unwrap();

        for (a, b) in x_values.iter().zip(result_values.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dropout_invalid_probability() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // p >= 1 is invalid
        let result = nn::dropout(&x, 1.0, true);
        assert!(result.is_err());

        // p < 0 is invalid
        let result = nn::dropout(&x, -0.1, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_dropout2d() {
        random::seed(123);

        // Input: (N, H, W, C) = (2, 4, 4, 8)
        let x = Array::ones::<f32>(&[2, 4, 4, 8]).unwrap();

        let result = nn::dropout2d(&x, 0.5, true).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![2, 4, 4, 8]);

        // Check that entire channels are dropped
        let values = result.to_vec::<f32>().unwrap();

        // For each batch and channel, all spatial locations should be the same
        // (either all zeros or all scaled)
        for n in 0..2 {
            for c in 0..8 {
                let base_idx = n * 4 * 4 * 8 + c;
                let first_val = values[base_idx];

                // Check all spatial positions for this channel
                for h in 0..4 {
                    for w in 0..4 {
                        let idx = n * 4 * 4 * 8 + h * 4 * 8 + w * 8 + c;
                        assert!(
                            (values[idx] - first_val).abs() < 1e-5,
                            "Channel values should be consistent"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_dropout2d_eval_mode() {
        let x = Array::ones::<f32>(&[2, 4, 4, 8]).unwrap();

        let result = nn::dropout2d(&x, 0.5, false).unwrap();
        result.eval();

        // All values should be 1.0 in eval mode
        let values = result.to_vec::<f32>().unwrap();
        for v in values.iter() {
            assert!((*v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dropout3d() {
        random::seed(456);

        // Input: (N, D, H, W, C) = (1, 2, 2, 2, 4)
        let x = Array::ones::<f32>(&[1, 2, 2, 2, 4]).unwrap();

        let result = nn::dropout3d(&x, 0.5, true).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![1, 2, 2, 2, 4]);
    }

    #[test]
    fn test_alpha_dropout() {
        random::seed(789);

        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();

        let result = nn::alpha_dropout(&x, 0.5, true).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![2, 4]);

        // Alpha dropout produces non-zero values (saturation values) where elements are dropped
        let values = result.to_vec::<f32>().unwrap();
        for v in values.iter() {
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn test_alpha_dropout_eval_mode() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = nn::alpha_dropout(&x, 0.5, false).unwrap();
        result.eval();

        let x_values = x.to_vec::<f32>().unwrap();
        let result_values = result.to_vec::<f32>().unwrap();

        for (a, b) in x_values.iter().zip(result_values.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ============================================================================
    // Attention Tests
    // ============================================================================

    #[test]
    fn test_scaled_dot_product_attention() {
        // Input: (batch, num_heads, seq_len, head_dim) = (2, 4, 8, 16)
        let q = Array::ones::<f32>(&[2, 4, 8, 16]).unwrap();
        let k = Array::ones::<f32>(&[2, 4, 8, 16]).unwrap();
        let v = Array::ones::<f32>(&[2, 4, 8, 16]).unwrap();

        let result = nn::scaled_dot_product_attention(
            &q, &k, &v,
            None,
            nn::AttentionMask::None,
            None,
        ).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![2, 4, 8, 16]);
    }

    #[test]
    fn test_scaled_dot_product_attention_causal() {
        // Input: (batch, num_heads, seq_len, head_dim) = (1, 2, 4, 8)
        let q = Array::ones::<f32>(&[1, 2, 4, 8]).unwrap();
        let k = Array::ones::<f32>(&[1, 2, 4, 8]).unwrap();
        let v = Array::ones::<f32>(&[1, 2, 4, 8]).unwrap();

        let result = nn::scaled_dot_product_attention(
            &q, &k, &v,
            None,
            nn::AttentionMask::Causal,
            None,
        ).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![1, 2, 4, 8]);

        // With causal mask, output should still be valid
        let values = result.to_vec::<f32>().unwrap();
        for v in values.iter() {
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn test_scaled_dot_product_attention_with_scale() {
        let q = Array::ones::<f32>(&[1, 1, 4, 8]).unwrap();
        let k = Array::ones::<f32>(&[1, 1, 4, 8]).unwrap();
        let v = Array::ones::<f32>(&[1, 1, 4, 8]).unwrap();

        // Custom scale
        let result = nn::scaled_dot_product_attention(
            &q, &k, &v,
            Some(0.5),
            nn::AttentionMask::None,
            None,
        ).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![1, 1, 4, 8]);
    }

    #[test]
    fn test_multi_head_attention_simple() {
        // Input: (batch, seq_len, embed_dim) = (2, 10, 64) with 8 heads
        let q = Array::ones::<f32>(&[2, 10, 64]).unwrap();
        let k = Array::ones::<f32>(&[2, 10, 64]).unwrap();
        let v = Array::ones::<f32>(&[2, 10, 64]).unwrap();

        let result = nn::multi_head_attention_simple(
            &q, &k, &v,
            8,  // num_heads
            nn::AttentionMask::None,
            None,
        ).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![2, 10, 64]);
    }

    #[test]
    fn test_multi_head_attention_simple_causal() {
        // Input: (batch, seq_len, embed_dim) = (1, 8, 32) with 4 heads
        let q = Array::ones::<f32>(&[1, 8, 32]).unwrap();
        let k = Array::ones::<f32>(&[1, 8, 32]).unwrap();
        let v = Array::ones::<f32>(&[1, 8, 32]).unwrap();

        let result = nn::multi_head_attention_simple(
            &q, &k, &v,
            4,  // num_heads
            nn::AttentionMask::Causal,
            None,
        ).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![1, 8, 32]);
    }

    #[test]
    fn test_multi_head_attention_with_weights() {
        // Input: (batch, seq_len, embed_dim) = (1, 4, 16) with 4 heads
        let q = Array::ones::<f32>(&[1, 4, 16]).unwrap();
        let k = Array::ones::<f32>(&[1, 4, 16]).unwrap();
        let v = Array::ones::<f32>(&[1, 4, 16]).unwrap();

        // Projection weights
        let w_q = random::glorot_uniform(&[16, 16], None).unwrap();
        let w_k = random::glorot_uniform(&[16, 16], None).unwrap();
        let w_v = random::glorot_uniform(&[16, 16], None).unwrap();
        let w_o = random::glorot_uniform(&[16, 16], None).unwrap();

        let result = nn::multi_head_attention(
            &q, &k, &v,
            4,  // num_heads
            &w_q, &w_k, &w_v, &w_o,
            nn::AttentionMask::None,
            None,
        ).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![1, 4, 16]);
    }

    #[test]
    fn test_multi_head_attention_invalid_heads() {
        // embed_dim=10 is not divisible by num_heads=3
        let q = Array::ones::<f32>(&[1, 4, 10]).unwrap();
        let k = Array::ones::<f32>(&[1, 4, 10]).unwrap();
        let v = Array::ones::<f32>(&[1, 4, 10]).unwrap();

        let result = nn::multi_head_attention_simple(
            &q, &k, &v,
            3,  // num_heads - doesn't divide evenly
            nn::AttentionMask::None,
            None,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_attention_mask_custom_requires_array() {
        let q = Array::ones::<f32>(&[1, 1, 4, 8]).unwrap();
        let k = Array::ones::<f32>(&[1, 1, 4, 8]).unwrap();
        let v = Array::ones::<f32>(&[1, 1, 4, 8]).unwrap();

        // Custom mask without array should fail
        let result = nn::scaled_dot_product_attention(
            &q, &k, &v,
            None,
            nn::AttentionMask::Custom,
            None,
        );

        assert!(result.is_err());
    }

    // ============================================================================
    // Embedding Tests
    // ============================================================================

    #[test]
    fn test_embedding() {
        // Create embedding table: 10 words, 4-dim embeddings
        let weight = Array::from_slice(
            &[
                1.0f32, 0.0, 0.0, 0.0,  // word 0
                0.0, 1.0, 0.0, 0.0,     // word 1
                0.0, 0.0, 1.0, 0.0,     // word 2
                0.0, 0.0, 0.0, 1.0,     // word 3
                1.0, 1.0, 0.0, 0.0,     // word 4
                0.0, 1.0, 1.0, 0.0,     // word 5
                0.0, 0.0, 1.0, 1.0,     // word 6
                1.0, 0.0, 0.0, 1.0,     // word 7
                1.0, 1.0, 1.0, 0.0,     // word 8
                0.0, 1.0, 1.0, 1.0,     // word 9
            ],
            &[10, 4],
        ).unwrap();

        // Look up words 0, 2, 5
        let indices = Array::from_slice(&[0i32, 2, 5], &[3]).unwrap();

        let embeddings = nn::embedding(&weight, &indices).unwrap();
        embeddings.eval();

        assert_eq!(embeddings.shape(), vec![3, 4]);

        let values = embeddings.to_vec::<f32>().unwrap();
        // Word 0: [1, 0, 0, 0]
        assert!((values[0] - 1.0).abs() < 1e-5);
        assert!((values[1] - 0.0).abs() < 1e-5);
        // Word 2: [0, 0, 1, 0]
        assert!((values[4 + 2] - 1.0).abs() < 1e-5);
        // Word 5: [0, 1, 1, 0]
        assert!((values[8 + 1] - 1.0).abs() < 1e-5);
        assert!((values[8 + 2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_2d_indices() {
        // Create embedding table
        let weight = random::normal::<f32>(&[100, 32], None).unwrap();

        // 2D indices (batch of sequences)
        let indices = Array::from_slice(
            &[1i32, 5, 10, 20, 30, 40],
            &[2, 3],  // batch=2, seq_len=3
        ).unwrap();

        let embeddings = nn::embedding(&weight, &indices).unwrap();
        embeddings.eval();

        assert_eq!(embeddings.shape(), vec![2, 3, 32]);
    }

    #[test]
    fn test_embedding_with_padding() {
        // Create embedding table
        let weight = Array::ones::<f32>(&[10, 4]).unwrap();

        // Indices with padding (padding_idx = 0)
        let indices = Array::from_slice(&[0i32, 1, 0, 2], &[4]).unwrap();

        let embeddings = nn::embedding_with_padding(&weight, &indices, 0).unwrap();
        embeddings.eval();

        assert_eq!(embeddings.shape(), vec![4, 4]);

        let values = embeddings.to_vec::<f32>().unwrap();
        // Indices 0 and 2 should be zeros (padding)
        assert!((values[0] - 0.0).abs() < 1e-5);
        assert!((values[1] - 0.0).abs() < 1e-5);
        // Index 1 should be ones
        assert!((values[4] - 1.0).abs() < 1e-5);
        // Index 2 (position 2) should be zeros
        assert!((values[8] - 0.0).abs() < 1e-5);
        // Index 3 should be ones
        assert!((values[12] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sinusoidal_positional_encoding() {
        let max_len = 100;
        let embed_dim = 64;

        let pe = nn::sinusoidal_positional_encoding(max_len, embed_dim).unwrap();
        pe.eval();

        assert_eq!(pe.shape(), vec![max_len, embed_dim]);

        // Values should be in [-1, 1] range
        let values = pe.to_vec::<f32>().unwrap();
        for v in values.iter() {
            assert!(*v >= -1.0 - 1e-5 && *v <= 1.0 + 1e-5);
        }
    }

    #[test]
    fn test_sinusoidal_encoding_odd_dim_error() {
        // Odd embed_dim should fail
        let result = nn::sinusoidal_positional_encoding(100, 63);
        assert!(result.is_err());
    }

    #[test]
    fn test_learned_positional_embedding() {
        let max_len = 50;
        let embed_dim = 128;

        let pe = nn::learned_positional_embedding(max_len, embed_dim).unwrap();
        pe.eval();

        assert_eq!(pe.shape(), vec![max_len, embed_dim]);
    }

    #[test]
    fn test_add_positional_encoding() {
        let batch = 2;
        let seq_len = 10;
        let embed_dim = 32;

        let embeddings = Array::ones::<f32>(&[batch, seq_len, embed_dim]).unwrap();
        let pe = nn::sinusoidal_positional_encoding(100, embed_dim).unwrap();

        let result = nn::add_positional_encoding(&embeddings, &pe).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![batch, seq_len, embed_dim]);

        // Result should be embeddings + pe (not just ones anymore)
        let values = result.to_vec::<f32>().unwrap();
        // At position 0, dim 0: sin(0) = 0, so result should be 1 + 0 = 1
        assert!((values[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_precompute_rope_frequencies() {
        let max_len = 64;
        let head_dim = 32;

        let (cos, sin) = nn::precompute_rope_frequencies(max_len, head_dim, 10000.0).unwrap();
        cos.eval();
        sin.eval();

        assert_eq!(cos.shape(), vec![max_len, head_dim / 2]);
        assert_eq!(sin.shape(), vec![max_len, head_dim / 2]);

        // Values should be in [-1, 1]
        let cos_vals = cos.to_vec::<f32>().unwrap();
        let sin_vals = sin.to_vec::<f32>().unwrap();
        for v in cos_vals.iter().chain(sin_vals.iter()) {
            assert!(*v >= -1.0 - 1e-5 && *v <= 1.0 + 1e-5);
        }
    }

    #[test]
    fn test_apply_rotary_embedding() {
        let batch = 1;
        let num_heads = 2;
        let seq_len = 8;
        let head_dim = 16;

        let x = Array::ones::<f32>(&[batch, num_heads, seq_len, head_dim]).unwrap();
        let (cos, sin) = nn::precompute_rope_frequencies(seq_len, head_dim, 10000.0).unwrap();

        let result = nn::apply_rotary_embedding(&x, &cos, &sin).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![batch, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_rope_odd_dim_error() {
        // Odd head_dim should fail
        let result = nn::precompute_rope_frequencies(64, 33, 10000.0);
        assert!(result.is_err());
    }

    // ========================================================================
    // More Activation Function Tests
    // ========================================================================

    #[test]
    fn test_gelu() {
        let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let result = nn::gelu(&x).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![5]);
        let values = result.to_vec::<f32>().unwrap();

        // GELU(0) = 0
        assert!((values[2]).abs() < 1e-5);
        // GELU is approximately identity for large positive values
        assert!((values[4] - 2.0).abs() < 0.1);
        // GELU is approximately 0 for large negative values
        assert!(values[0].abs() < 0.1);
    }

    #[test]
    fn test_gelu_approx() {
        let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let result = nn::gelu_approx(&x).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![5]);
        let values = result.to_vec::<f32>().unwrap();

        // Similar behavior to exact GELU
        assert!((values[2]).abs() < 1e-5);
        assert!((values[4] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_elu() {
        let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let alpha = 1.0;
        let result = nn::elu(&x, alpha).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![5]);
        let values = result.to_vec::<f32>().unwrap();

        // ELU(0) = 0
        assert!((values[2]).abs() < 1e-5);
        // ELU(x) = x for x > 0
        assert!((values[3] - 1.0).abs() < 1e-5);
        assert!((values[4] - 2.0).abs() < 1e-5);
        // ELU(x) = alpha * (exp(x) - 1) for x < 0
        // ELU(-1) = 1.0 * (exp(-1) - 1) ≈ -0.632
        assert!((values[1] - (-0.632)).abs() < 0.01);
    }

    #[test]
    fn test_selu() {
        let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let result = nn::selu(&x).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![5]);
        let values = result.to_vec::<f32>().unwrap();

        // SELU(0) = 0
        assert!((values[2]).abs() < 1e-5);
        // SELU(x) = scale * x for x > 0
        let scale = 1.0507009873554805f32;
        assert!((values[3] - scale * 1.0).abs() < 1e-5);
        assert!((values[4] - scale * 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_celu() {
        let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let alpha = 1.0;
        let result = nn::celu(&x, alpha).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![5]);
        let values = result.to_vec::<f32>().unwrap();

        // CELU(0) = 0
        assert!((values[2]).abs() < 1e-5);
        // CELU(x) = x for x > 0
        assert!((values[3] - 1.0).abs() < 1e-5);
        assert!((values[4] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_mish() {
        let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let result = nn::mish(&x).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![5]);
        let values = result.to_vec::<f32>().unwrap();

        // Mish(0) = 0 * tanh(softplus(0)) = 0
        assert!((values[2]).abs() < 1e-5);
        // Mish approaches x for large positive x
        assert!((values[4] - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_hardswish() {
        let x = Array::from_slice(&[-4.0f32, -3.0, 0.0, 3.0, 4.0], &[5]).unwrap();
        let result = nn::hardswish(&x).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![5]);
        let values = result.to_vec::<f32>().unwrap();

        // Hardswish(x) = 0 for x <= -3
        assert!((values[0]).abs() < 1e-5);
        assert!((values[1]).abs() < 1e-5);
        // Hardswish(0) = 0 * (0 + 3) / 6 = 0
        assert!((values[2]).abs() < 1e-5);
        // Hardswish(x) = x for x >= 3
        assert!((values[3] - 3.0).abs() < 1e-5);
        assert!((values[4] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_hardsigmoid() {
        let x = Array::from_slice(&[-4.0f32, -3.0, 0.0, 3.0, 4.0], &[5]).unwrap();
        let result = nn::hardsigmoid(&x).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![5]);
        let values = result.to_vec::<f32>().unwrap();

        // Hardsigmoid(x) = 0 for x <= -3
        assert!((values[0]).abs() < 1e-5);
        assert!((values[1]).abs() < 1e-5);
        // Hardsigmoid(0) = 0.5
        assert!((values[2] - 0.5).abs() < 1e-5);
        // Hardsigmoid(x) = 1 for x >= 3
        assert!((values[3] - 1.0).abs() < 1e-5);
        assert!((values[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hardtanh() {
        let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let result = nn::hardtanh(&x, -1.0, 1.0).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![5]);
        let values = result.to_vec::<f32>().unwrap();

        // Values should be clipped to [-1, 1]
        assert!((values[0] - (-1.0)).abs() < 1e-5);  // -2 -> -1
        assert!((values[1] - (-1.0)).abs() < 1e-5);  // -1 -> -1
        assert!((values[2]).abs() < 1e-5);           // 0 -> 0
        assert!((values[3] - 1.0).abs() < 1e-5);     // 1 -> 1
        assert!((values[4] - 1.0).abs() < 1e-5);     // 2 -> 1
    }

    #[test]
    fn test_log_softmax() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let result = nn::log_softmax(&x, 0).unwrap();
        result.eval();

        assert_eq!(result.shape(), vec![3]);
        let values = result.to_vec::<f32>().unwrap();

        // log_softmax values should be negative (log of probabilities)
        for v in &values {
            assert!(*v < 0.0);
        }

        // exp(log_softmax) should sum to 1
        let sum: f32 = values.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_2d() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let result = nn::log_softmax(&x, -1).unwrap();  // Along last axis
        result.eval();

        assert_eq!(result.shape(), vec![2, 3]);
        let values = result.to_vec::<f32>().unwrap();

        // Each row should have exp(log_softmax) sum to 1
        let sum1: f32 = values[0..3].iter().map(|v| v.exp()).sum();
        let sum2: f32 = values[3..6].iter().map(|v| v.exp()).sum();
        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_glu() {
        // GLU splits input in half along axis and applies sigmoid gate
        let _x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        // GLU needs even size along the split axis
        let x2 = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = nn::glu(&x2, -1).unwrap();  // Split along last axis (size 2 -> size 1)
        result.eval();

        // Output shape should be half the input along the split axis
        assert_eq!(result.shape(), vec![2, 1]);
    }

    #[test]
    fn test_glu_batch() {
        // Batch of vectors, split along feature dimension
        let x = Array::from_slice(&[
            1.0f32, 2.0, 3.0, 4.0,  // First sample
            5.0, 6.0, 7.0, 8.0,     // Second sample
        ], &[2, 4]).unwrap();

        let result = nn::glu(&x, -1).unwrap();
        result.eval();

        // Output should be [2, 2]
        assert_eq!(result.shape(), vec![2, 2]);
    }

    // ========================================================================
    // Loss Function Tests
    // ========================================================================

    #[test]
    fn test_mse_loss() {
        let predictions = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let targets = Array::from_slice(&[1.5f32, 2.0, 2.5], &[3]).unwrap();

        let loss = nn::mse_loss(&predictions, &targets, "mean").unwrap();
        loss.eval();

        // MSE = mean((1-1.5)^2 + (2-2)^2 + (3-2.5)^2) = mean(0.25 + 0 + 0.25) = 0.167
        let value = loss.to_vec::<f32>().unwrap()[0];
        assert!((value - 0.1667).abs() < 0.01);
    }

    #[test]
    fn test_l1_loss() {
        let predictions = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let targets = Array::from_slice(&[1.5f32, 2.0, 2.5], &[3]).unwrap();

        let loss = nn::l1_loss(&predictions, &targets, "mean").unwrap();
        loss.eval();

        // MAE = mean(|1-1.5| + |2-2| + |3-2.5|) = mean(0.5 + 0 + 0.5) = 0.333
        let value = loss.to_vec::<f32>().unwrap()[0];
        assert!((value - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let predictions = Array::from_slice(&[0.9f32, 0.1, 0.8], &[3]).unwrap();
        let targets = Array::from_slice(&[1.0f32, 0.0, 1.0], &[3]).unwrap();

        let loss = nn::binary_cross_entropy(&predictions, &targets, "mean").unwrap();
        loss.eval();

        // Low loss because predictions are close to targets
        let value = loss.to_vec::<f32>().unwrap()[0];
        assert!(value > 0.0);
        assert!(value < 1.0);  // Should be relatively small
    }

    #[test]
    fn test_huber_loss() {
        let predictions = Array::from_slice(&[1.0f32, 2.0, 10.0], &[3]).unwrap();
        let targets = Array::from_slice(&[1.5f32, 2.0, 2.0], &[3]).unwrap();

        let loss = nn::huber_loss(&predictions, &targets, 1.0, "mean").unwrap();
        loss.eval();

        let value = loss.to_vec::<f32>().unwrap()[0];
        assert!(value > 0.0);
    }

    #[test]
    fn test_binary_cross_entropy_with_logits() {
        // Logits close to correct predictions
        let logits = Array::from_slice(&[2.0f32, -2.0, 2.0], &[3]).unwrap();
        let targets = Array::from_slice(&[1.0f32, 0.0, 1.0], &[3]).unwrap();

        let loss = nn::binary_cross_entropy_with_logits(&logits, &targets, "mean").unwrap();
        loss.eval();

        let value = loss.to_vec::<f32>().unwrap()[0];
        assert!(value > 0.0);
        assert!(value < 1.0);  // Low loss since predictions match
    }

    #[test]
    fn test_cross_entropy_loss() {
        // One-hot targets
        let predictions = Array::from_slice(&[
            0.7f32, 0.2, 0.1,  // Class 0 predicted
            0.1, 0.8, 0.1,     // Class 1 predicted
        ], &[2, 3]).unwrap();
        let targets = Array::from_slice(&[
            1.0f32, 0.0, 0.0,  // True class 0
            0.0, 1.0, 0.0,     // True class 1
        ], &[2, 3]).unwrap();

        let loss = nn::cross_entropy_loss(&predictions, &targets, "mean").unwrap();
        loss.eval();

        let value = loss.to_vec::<f32>().unwrap()[0];
        assert!(value > 0.0);
        assert!(value < 1.0);  // Good predictions -> low loss
    }

    #[test]
    fn test_hinge_loss() {
        // SVM-style loss with y in {-1, +1}
        let predictions = Array::from_slice(&[0.5f32, -0.5, 1.5], &[3]).unwrap();
        let targets = Array::from_slice(&[1.0f32, -1.0, 1.0], &[3]).unwrap();

        let loss = nn::hinge_loss(&predictions, &targets, 1.0, "mean").unwrap();
        loss.eval();

        let value = loss.to_vec::<f32>().unwrap()[0];
        // Hinge loss = max(0, 1 - y*pred)
        // For (0.5, 1): max(0, 1-0.5) = 0.5
        // For (-0.5, -1): max(0, 1-0.5) = 0.5
        // For (1.5, 1): max(0, 1-1.5) = 0
        // Mean = 0.333
        assert!((value - 0.333).abs() < 0.1);
    }

    #[test]
    fn test_triplet_margin_loss() {
        let anchor = Array::from_slice(&[1.0f32, 0.0], &[1, 2]).unwrap();
        let positive = Array::from_slice(&[1.1f32, 0.1], &[1, 2]).unwrap();
        let negative = Array::from_slice(&[0.0f32, 1.0], &[1, 2]).unwrap();

        let loss = nn::triplet_margin_loss(&anchor, &positive, &negative, 1.0, "mean").unwrap();
        loss.eval();

        // Positive is closer to anchor than negative, so loss should be low
        let value = loss.to_vec::<f32>().unwrap()[0];
        assert!(value >= 0.0);
    }

    #[test]
    fn test_loss_reduction_none() {
        let predictions = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let targets = Array::from_slice(&[1.5f32, 2.0, 2.5], &[3]).unwrap();

        let loss = nn::mse_loss(&predictions, &targets, "none").unwrap();
        loss.eval();

        // With reduction="none", output should have same shape as input
        assert_eq!(loss.shape(), vec![3]);
    }

    #[test]
    fn test_loss_reduction_sum() {
        let predictions = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let targets = Array::from_slice(&[1.5f32, 2.0, 2.5], &[3]).unwrap();

        let loss = nn::mse_loss(&predictions, &targets, "sum").unwrap();
        loss.eval();

        // Sum of (0.25 + 0 + 0.25) = 0.5
        let value = loss.to_vec::<f32>().unwrap()[0];
        assert!((value - 0.5).abs() < 0.01);
    }

    // ========================================================================
    // Optimizer tests
    // ========================================================================

    #[test]
    fn test_sgd_step() {
        let param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap();

        let new_param = nn::sgd_step(&param, &grad, 0.1).unwrap();
        new_param.eval();

        let values = new_param.to_vec::<f32>().unwrap();
        // new_param = param - lr * grad
        // = [1.0 - 0.01, 2.0 - 0.02, 3.0 - 0.03]
        assert!((values[0] - 0.99).abs() < 0.001);
        assert!((values[1] - 1.98).abs() < 0.001);
        assert!((values[2] - 2.97).abs() < 0.001);
    }

    #[test]
    fn test_sgd_momentum_step() {
        let param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap();
        let velocity = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();

        let (new_param, new_velocity) = nn::sgd_momentum_step(&param, &grad, &velocity, 0.1, 0.9).unwrap();
        new_param.eval();
        new_velocity.eval();

        // First step: velocity = 0.9 * 0 + grad = grad
        let vel_values = new_velocity.to_vec::<f32>().unwrap();
        assert!((vel_values[0] - 0.1).abs() < 0.001);

        // param = param - lr * velocity
        let param_values = new_param.to_vec::<f32>().unwrap();
        assert!((param_values[0] - 0.99).abs() < 0.001);
    }

    #[test]
    fn test_sgd_weight_decay_step() {
        let param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap();
        let velocity = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();

        let (new_param, new_velocity) = nn::sgd_weight_decay_step(
            &param, &grad, &velocity, 0.1, 0.9, 0.01
        ).unwrap();
        new_param.eval();
        new_velocity.eval();

        // With weight decay, gradients are augmented
        let param_values = new_param.to_vec::<f32>().unwrap();
        // Values should be less than without weight decay
        assert!(param_values[0] < 1.0);
        assert!(param_values[1] < 2.0);
        assert!(param_values[2] < 3.0);
    }

    #[test]
    fn test_adam_step() {
        let param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap();
        let m = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();
        let v = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();

        let (new_param, new_m, new_v) = nn::adam_step(
            &param, &grad, &m, &v, 0.001, 0.9, 0.999, 1e-8, 1
        ).unwrap();
        new_param.eval();
        new_m.eval();
        new_v.eval();

        // Check that parameters moved
        let param_values = new_param.to_vec::<f32>().unwrap();
        assert!(param_values[0] < 1.0);
        assert!(param_values[1] < 2.0);
        assert!(param_values[2] < 3.0);

        // Check that moments are updated
        let m_values = new_m.to_vec::<f32>().unwrap();
        assert!(m_values[0] > 0.0);

        let v_values = new_v.to_vec::<f32>().unwrap();
        assert!(v_values[0] > 0.0);
    }

    #[test]
    fn test_adamw_step() {
        let param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap();
        let m = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();
        let v = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();

        let (new_param, new_m, new_v) = nn::adamw_step(
            &param, &grad, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1
        ).unwrap();
        new_param.eval();
        new_m.eval();
        new_v.eval();

        // Check that parameters moved (should decrease more than adam due to weight decay)
        let param_values = new_param.to_vec::<f32>().unwrap();
        assert!(param_values[0] < 1.0);
        assert!(param_values[1] < 2.0);
        assert!(param_values[2] < 3.0);
    }

    #[test]
    fn test_rmsprop_step() {
        let param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap();
        let v = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();

        let (new_param, new_v) = nn::rmsprop_step(&param, &grad, &v, 0.01, 0.9, 1e-8).unwrap();
        new_param.eval();
        new_v.eval();

        // Check that parameters moved
        let param_values = new_param.to_vec::<f32>().unwrap();
        assert!(param_values[0] < 1.0);
        assert!(param_values[1] < 2.0);
        assert!(param_values[2] < 3.0);

        // Check velocity is updated
        let v_values = new_v.to_vec::<f32>().unwrap();
        assert!(v_values[0] > 0.0);
    }

    #[test]
    fn test_adagrad_step() {
        let param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad = Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap();
        let accumulated = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();

        let (new_param, new_accumulated) = nn::adagrad_step(
            &param, &grad, &accumulated, 0.01, 1e-8
        ).unwrap();
        new_param.eval();
        new_accumulated.eval();

        // Check that parameters moved
        let param_values = new_param.to_vec::<f32>().unwrap();
        assert!(param_values[0] < 1.0);
        assert!(param_values[1] < 2.0);
        assert!(param_values[2] < 3.0);

        // Check accumulated is updated (should be grad^2)
        let acc_values = new_accumulated.to_vec::<f32>().unwrap();
        assert!((acc_values[0] - 0.01).abs() < 0.001); // 0.1^2 = 0.01
    }

    #[test]
    fn test_init_optimizer_state() {
        let param = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();

        let state = nn::init_optimizer_state(&param).unwrap();
        state.eval();

        // State should be zeros with same shape
        assert_eq!(state.shape(), param.shape());
        let values = state.to_vec::<f32>().unwrap();
        assert!((values[0]).abs() < 0.001);
        assert!((values[1]).abs() < 0.001);
        assert!((values[2]).abs() < 0.001);
    }

    #[test]
    fn test_optimizer_multiple_steps() {
        // Test that multiple optimizer steps converge towards minimum
        let mut param = Array::from_slice(&[5.0f32], &[1]).unwrap();
        let mut m = Array::from_slice(&[0.0f32], &[1]).unwrap();
        let mut v = Array::from_slice(&[0.0f32], &[1]).unwrap();

        // Simple quadratic: f(x) = x^2, gradient = 2x
        // Minimum at x = 0
        for t in 1..=100 {
            // Gradient of x^2 is 2x
            let grad = &param * &Array::from_float(2.0);
            let result = nn::adam_step(&param, &grad, &m, &v, 0.1, 0.9, 0.999, 1e-8, t).unwrap();
            param = result.0;
            m = result.1;
            v = result.2;
            param.eval();
            m.eval();
            v.eval();
        }

        let final_value = param.to_vec::<f32>().unwrap()[0];
        // Should be close to 0 after 100 steps
        assert!(final_value.abs() < 0.1);
    }

    // ========================================================================
    // Auto-diff tests
    // ========================================================================

    #[test]
    fn test_vjp_simple() {
        // f(x) = x^2, df/dx = 2x
        // At x = 3, df/dx = 6
        let x = Array::from_float(3.0);

        let (outputs, grads) = transforms::vjp(
            |inputs| {
                let x = &inputs[0];
                vec![x * x]
            },
            &[x],
            &[Array::from_float(1.0)], // cotangent = 1 for scalar gradient
        ).unwrap();

        outputs[0].eval();
        grads[0].eval();

        let output_val = outputs[0].to_vec::<f32>().unwrap()[0];
        let grad_val = grads[0].to_vec::<f32>().unwrap()[0];

        assert!((output_val - 9.0).abs() < 0.001); // 3^2 = 9
        assert!((grad_val - 6.0).abs() < 0.001);   // 2*3 = 6
    }

    #[test]
    fn test_vjp_multi_input() {
        // f(x, y) = x * y
        // df/dx = y, df/dy = x
        // At (x=2, y=3): df/dx = 3, df/dy = 2
        let x = Array::from_float(2.0);
        let y = Array::from_float(3.0);

        let (outputs, grads) = transforms::vjp(
            |inputs| {
                let x = &inputs[0];
                let y = &inputs[1];
                vec![x * y]
            },
            &[x, y],
            &[Array::from_float(1.0)],
        ).unwrap();

        outputs[0].eval();
        grads[0].eval();
        grads[1].eval();

        let output_val = outputs[0].to_vec::<f32>().unwrap()[0];
        assert!((output_val - 6.0).abs() < 0.001); // 2 * 3 = 6

        let grad_x = grads[0].to_vec::<f32>().unwrap()[0];
        let grad_y = grads[1].to_vec::<f32>().unwrap()[0];
        assert!((grad_x - 3.0).abs() < 0.001); // df/dx = y = 3
        assert!((grad_y - 2.0).abs() < 0.001); // df/dy = x = 2
    }

    #[test]
    fn test_jvp_simple() {
        // f(x) = x^2, df/dx = 2x
        // JVP computes df/dx * v where v is tangent
        // At x = 3 with tangent = 1: JVP = 2*3*1 = 6
        let x = Array::from_float(3.0);

        let (outputs, jvps) = transforms::jvp(
            |inputs| {
                let x = &inputs[0];
                vec![x * x]
            },
            &[x],
            &[Array::from_float(1.0)], // tangent = 1
        ).unwrap();

        outputs[0].eval();
        jvps[0].eval();

        let output_val = outputs[0].to_vec::<f32>().unwrap()[0];
        let jvp_val = jvps[0].to_vec::<f32>().unwrap()[0];

        assert!((output_val - 9.0).abs() < 0.001); // 3^2 = 9
        assert!((jvp_val - 6.0).abs() < 0.001);    // 2*3 = 6
    }

    #[test]
    fn test_value_and_grad_simple() {
        // f(x) = x^2
        let x = Array::from_float(3.0);

        let (values, grads) = transforms::value_and_grad(
            |inputs| {
                let x = &inputs[0];
                vec![x * x]
            },
            &[x],
            &[0], // differentiate w.r.t. first argument
        ).unwrap();

        values[0].eval();
        grads[0].eval();

        let value = values[0].to_vec::<f32>().unwrap()[0];
        let grad = grads[0].to_vec::<f32>().unwrap()[0];

        assert!((value - 9.0).abs() < 0.001); // 3^2 = 9
        assert!((grad - 6.0).abs() < 0.001);  // 2*3 = 6
    }

    #[test]
    fn test_grad_simple() {
        // f(x) = x^2, grad at x=4 should be 8
        let x = Array::from_float(4.0);

        let grads = transforms::grad(
            |inputs| {
                let x = &inputs[0];
                vec![x * x]
            },
            &[x],
            &[0],
        ).unwrap();

        grads[0].eval();
        let grad = grads[0].to_vec::<f32>().unwrap()[0];
        assert!((grad - 8.0).abs() < 0.001); // 2*4 = 8
    }

    #[test]
    fn test_grad_single() {
        // f(x) = x^3, f'(x) = 3x^2
        // At x = 2: f'(2) = 3*4 = 12
        let x = Array::from_float(2.0);

        let grad = transforms::grad_single(
            |x| x * x * x,
            &x,
        ).unwrap();

        grad.eval();
        let grad_val = grad.to_vec::<f32>().unwrap()[0];
        assert!((grad_val - 12.0).abs() < 0.001);
    }

    #[test]
    fn test_value_and_grad_single() {
        // f(x) = x^2 + 2x
        // f'(x) = 2x + 2
        // At x = 3: f(3) = 9 + 6 = 15, f'(3) = 6 + 2 = 8
        let x = Array::from_float(3.0);

        let (value, grad) = transforms::value_and_grad_single(
            |x| {
                let x_squared = x * x;
                let two_x = x * &Array::from_float(2.0);
                &x_squared + &two_x
            },
            &x,
        ).unwrap();

        value.eval();
        grad.eval();

        let val = value.to_vec::<f32>().unwrap()[0];
        let g = grad.to_vec::<f32>().unwrap()[0];

        assert!((val - 15.0).abs() < 0.001);
        assert!((g - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_stop_gradient() {
        let x = Array::from_float(3.0);
        let y = transforms::stop_gradient(&x).unwrap();
        y.eval();

        let val = y.to_vec::<f32>().unwrap()[0];
        assert!((val - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_vjp_with_array() {
        // f(x) = sum(x^2) for vector x
        // grad = 2x
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();

        let (outputs, grads) = transforms::vjp(
            |inputs| {
                let x = &inputs[0];
                let x_squared = x * x;
                vec![x_squared.sum_all(false).unwrap()]
            },
            &[x],
            &[Array::from_float(1.0)],
        ).unwrap();

        outputs[0].eval();
        grads[0].eval();

        let output_val = outputs[0].to_vec::<f32>().unwrap()[0];
        // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
        assert!((output_val - 14.0).abs() < 0.001);

        let grad_vals = grads[0].to_vec::<f32>().unwrap();
        // grad = 2x = [2, 4, 6]
        assert!((grad_vals[0] - 2.0).abs() < 0.001);
        assert!((grad_vals[1] - 4.0).abs() < 0.001);
        assert!((grad_vals[2] - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_autodiff_chain_rule() {
        // f(x) = sin(x^2)
        // f'(x) = cos(x^2) * 2x
        // At x = 1: f'(1) = cos(1) * 2 ≈ 0.5403 * 2 ≈ 1.0806
        let x = Array::from_float(1.0);

        let (values, grads) = transforms::value_and_grad(
            |inputs| {
                let x = &inputs[0];
                let x_sq = x * x;
                vec![x_sq.sin().unwrap()]
            },
            &[x],
            &[0],
        ).unwrap();

        values[0].eval();
        grads[0].eval();

        let val = values[0].to_vec::<f32>().unwrap()[0];
        let grad = grads[0].to_vec::<f32>().unwrap()[0];

        // sin(1) ≈ 0.8414
        assert!((val - 0.8414).abs() < 0.01);
        // cos(1) * 2 ≈ 1.0806
        assert!((grad - 1.0806).abs() < 0.01);
    }

    // ========================================================================
    // Stateful optimizer tests
    // ========================================================================

    #[test]
    fn test_sgd_optimizer_struct() {
        use nn::{SGD, Optimizer};

        let mut optimizer = SGD::new(0.1);

        let params = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap()];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap()];

        let new_params = optimizer.step(&params, &grads).unwrap();
        new_params[0].eval();

        let values = new_params[0].to_vec::<f32>().unwrap();
        // param = param - lr * grad = [1.0 - 0.01, 2.0 - 0.02, 3.0 - 0.03]
        assert!((values[0] - 0.99).abs() < 0.001);
        assert!((values[1] - 1.98).abs() < 0.001);
        assert!((values[2] - 2.97).abs() < 0.001);

        assert_eq!(optimizer.step_count(), 1);
        assert!((optimizer.learning_rate() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_sgd_with_momentum() {
        use nn::{SGD, Optimizer};

        let mut optimizer = SGD::new(0.1).momentum(0.9);

        let params = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap()];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap()];

        // First step
        let new_params = optimizer.step(&params, &grads).unwrap();
        new_params[0].eval();

        // Second step - momentum should accumulate
        let grads2 = vec![Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap()];
        let new_params2 = optimizer.step(&new_params, &grads2).unwrap();
        new_params2[0].eval();

        assert_eq!(optimizer.step_count(), 2);
    }

    #[test]
    fn test_adam_optimizer_struct() {
        use nn::{Adam, Optimizer};

        let mut optimizer = Adam::new(0.001)
            .betas(0.9, 0.999)
            .eps(1e-8);

        let params = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap()];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap()];

        let new_params = optimizer.step(&params, &grads).unwrap();
        new_params[0].eval();

        let values = new_params[0].to_vec::<f32>().unwrap();
        // Parameters should decrease since gradients are positive
        assert!(values[0] < 1.0);
        assert!(values[1] < 2.0);
        assert!(values[2] < 3.0);

        assert_eq!(optimizer.step_count(), 1);
    }

    #[test]
    fn test_adam_convergence() {
        use nn::{Adam, Optimizer};

        let mut optimizer = Adam::new(0.1);

        // Simple quadratic minimization: f(x) = x^2, minimum at x = 0
        let mut params = vec![Array::from_slice(&[5.0f32], &[1]).unwrap()];

        for _ in 0..100 {
            // Gradient of x^2 is 2x
            let x_val = params[0].to_vec::<f32>().unwrap()[0];
            let grads = vec![Array::from_slice(&[2.0 * x_val], &[1]).unwrap()];

            params = optimizer.step(&params, &grads).unwrap();
            params[0].eval();
        }

        let final_val = params[0].to_vec::<f32>().unwrap()[0];
        // Should converge close to 0
        assert!(final_val.abs() < 0.1);
    }

    #[test]
    fn test_adamw_optimizer_struct() {
        use nn::{AdamW, Optimizer};

        let mut optimizer = AdamW::new(0.001)
            .weight_decay(0.01);

        let params = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap()];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap()];

        let new_params = optimizer.step(&params, &grads).unwrap();
        new_params[0].eval();

        let values = new_params[0].to_vec::<f32>().unwrap();
        // Parameters should decrease (due to both gradient and weight decay)
        assert!(values[0] < 1.0);
        assert!(values[1] < 2.0);
        assert!(values[2] < 3.0);
    }

    #[test]
    fn test_rmsprop_optimizer_struct() {
        use nn::{RMSprop, Optimizer};

        let mut optimizer = RMSprop::new(0.01)
            .alpha(0.99)
            .eps(1e-8);

        let params = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap()];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap()];

        let new_params = optimizer.step(&params, &grads).unwrap();
        new_params[0].eval();

        let values = new_params[0].to_vec::<f32>().unwrap();
        assert!(values[0] < 1.0);
        assert!(values[1] < 2.0);
        assert!(values[2] < 3.0);
    }

    #[test]
    fn test_optimizer_reset() {
        use nn::{Adam, Optimizer};

        let mut optimizer = Adam::new(0.001);

        let params = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap()];
        let grads = vec![Array::from_slice(&[0.1f32, 0.2, 0.3], &[3]).unwrap()];

        let _ = optimizer.step(&params, &grads).unwrap();
        assert_eq!(optimizer.step_count(), 1);

        optimizer.reset();
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_optimizer_set_learning_rate() {
        use nn::{SGD, Optimizer};

        let mut optimizer = SGD::new(0.1);
        assert!((optimizer.learning_rate() - 0.1).abs() < 0.001);

        optimizer.set_learning_rate(0.01);
        assert!((optimizer.learning_rate() - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_optimizer_multiple_params() {
        use nn::{Adam, Optimizer};

        let mut optimizer = Adam::new(0.001);

        // Multiple parameter tensors (like in a real neural network)
        let params = vec![
            Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap(),    // weights
            Array::from_slice(&[0.5f32], &[1]).unwrap(),         // bias
        ];
        let grads = vec![
            Array::from_slice(&[0.1f32, 0.2], &[2]).unwrap(),
            Array::from_slice(&[0.05f32], &[1]).unwrap(),
        ];

        let new_params = optimizer.step(&params, &grads).unwrap();
        new_params[0].eval();
        new_params[1].eval();

        // Both parameters should be updated
        assert_eq!(new_params.len(), 2);
        let w_values = new_params[0].to_vec::<f32>().unwrap();
        let b_values = new_params[1].to_vec::<f32>().unwrap();

        assert!(w_values[0] < 1.0);
        assert!(b_values[0] < 0.5);
    }

    // ============================================================================
    // Llama Model Tests
    // ============================================================================

    #[test]
    fn test_llama_config() {
        use nn::LlamaConfig;

        // Test default config
        let config = LlamaConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);

        // Test Llama 2 7B config
        let config = LlamaConfig::llama2_7b();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.max_position_embeddings, 4096);
        assert_eq!(config.head_dim(), 128);  // 4096 / 32 = 128

        // Test Llama 3 8B config (with GQA)
        let config = LlamaConfig::llama3_8b();
        assert_eq!(config.num_key_value_heads, 8);  // GQA
        assert_eq!(config.vocab_size, 128256);

        // Test builder pattern
        let config = LlamaConfig::new()
            .vocab_size(50000)
            .hidden_size(512)
            .num_hidden_layers(4);
        assert_eq!(config.vocab_size, 50000);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_hidden_layers, 4);
    }

    #[test]
    fn test_swiglu() {
        use nn::swiglu;

        // Input: (batch, seq, hidden)
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        // Gate projection: (hidden, intermediate)
        let gate_proj = Array::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2]).unwrap();
        // Up projection: (hidden, intermediate)
        let up_proj = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]).unwrap();

        let result = swiglu(&x, &gate_proj, &up_proj).unwrap();
        result.eval();

        // Output shape should be (batch, seq, intermediate)
        assert_eq!(result.shape(), vec![1, 2, 2]);

        let values = result.to_vec::<f32>().unwrap();
        // SwiGLU = swish(x @ gate) * (x @ up)
        // swish(x) = x * sigmoid(x)
        assert!(values.iter().all(|&v| v > 0.0));  // All positive due to swish
    }

    #[test]
    fn test_llama_feedforward() {
        use nn::llama_feedforward;

        // Small dimensions for testing
        let batch = 1;
        let seq_len = 2;
        let hidden = 4;
        let intermediate = 8;

        // Input: (batch, seq, hidden)
        let x = Array::ones::<f32>(&[batch, seq_len, hidden]).unwrap();
        // Gate projection: (hidden, intermediate)
        let gate_proj = random::normal_with_params::<f32>(&[hidden, intermediate], 0.0, 0.1, None).unwrap();
        // Up projection: (hidden, intermediate)
        let up_proj = random::normal_with_params::<f32>(&[hidden, intermediate], 0.0, 0.1, None).unwrap();
        // Down projection: (intermediate, hidden)
        let down_proj = random::normal_with_params::<f32>(&[intermediate, hidden], 0.0, 0.1, None).unwrap();

        let result = llama_feedforward(&x, &gate_proj, &up_proj, &down_proj).unwrap();
        result.eval();

        // Output should have same shape as input
        assert_eq!(result.shape(), vec![batch, seq_len, hidden]);
    }

    #[test]
    fn test_repeat_kv() {
        // Test GQA expansion: repeat K,V heads to match Q heads
        // Input: (batch, num_kv_heads, seq_len, head_dim)
        let kv = Array::from_slice(&[
            1.0f32, 2.0, 3.0, 4.0,  // head 0
            5.0, 6.0, 7.0, 8.0,     // head 1
        ], &[1, 2, 1, 4]).unwrap();

        // Expand dims and broadcast
        let expanded = kv.expand_dims(2).unwrap();
        expanded.eval();
        assert_eq!(expanded.shape(), vec![1, 2, 1, 1, 4]);
    }

    #[test]
    fn test_llama_causal_mask() {
        use nn::create_causal_mask;

        let mask = create_causal_mask(4).unwrap();
        mask.eval();

        // Should be (4, 4) with upper triangle being -inf
        assert_eq!(mask.shape(), vec![4, 4]);

        let values = mask.to_vec::<f32>().unwrap();
        // Diagonal and below should be 0
        assert_eq!(values[0], 0.0);   // [0,0]
        assert_eq!(values[4], 0.0);   // [1,0]
        assert_eq!(values[5], 0.0);   // [1,1]
        // Above diagonal should be -inf
        assert!(values[1].is_infinite() && values[1] < 0.0);  // [0,1]
        assert!(values[2].is_infinite() && values[2] < 0.0);  // [0,2]
    }

    #[test]
    fn test_transpose_axes() {
        // Test the new transpose_axes method
        let x = Array::from_slice(&[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ], &[2, 3, 2]).unwrap();

        // Transpose axes [0, 2, 1] swaps axes 1 and 2
        let transposed = x.transpose_axes(&[0, 2, 1]).unwrap();
        transposed.eval();

        assert_eq!(transposed.shape(), vec![2, 2, 3]);
    }

    // ============================================================================
    // Comprehensive Array Operation Tests
    // ============================================================================

    #[test]
    fn test_array_division() {
        let a = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4]).unwrap();
        let b = Array::from_slice(&[2.0f32, 4.0, 5.0, 8.0], &[4]).unwrap();

        let result = &a / &b;
        result.eval();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_array_negation() {
        let a = Array::from_slice(&[1.0f32, -2.0, 3.0, -4.0], &[4]).unwrap();
        let result = -&a;
        result.eval();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn test_array_scalar_broadcast() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let scalar = Array::from_float(10.0);

        // Array + scalar
        let result = &a + &scalar;
        result.eval();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![11.0, 12.0, 13.0]);

        // Array * scalar
        let result = &a * &scalar;
        result.eval();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_array_broadcast_2d() {
        // (2, 3) + (1, 3) broadcasting
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Array::from_slice(&[10.0f32, 20.0, 30.0], &[1, 3]).unwrap();

        let result = &a + &b;
        result.eval();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_array_power() {
        // Test x^2 using x * x
        let a = Array::from_slice(&[2.0f32, 3.0, 4.0], &[3]).unwrap();

        let result = &a * &a;
        result.eval();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_array_abs() {
        let a = Array::from_slice(&[-3.0f32, -2.0, -1.0, 0.0, 1.0, 2.0], &[6]).unwrap();
        let result = a.abs().unwrap();
        result.eval();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![3.0, 2.0, 1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_array_sqrt() {
        let a = Array::from_slice(&[1.0f32, 4.0, 9.0, 16.0], &[4]).unwrap();
        let result = a.sqrt().unwrap();
        result.eval();
        assert_eq!(result.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_array_exp_log() {
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0], &[3]).unwrap();

        // exp
        let exp_result = a.exp().unwrap();
        exp_result.eval();
        let exp_vals = exp_result.to_vec::<f32>().unwrap();
        assert!((exp_vals[0] - 1.0).abs() < 1e-5);
        assert!((exp_vals[1] - std::f32::consts::E).abs() < 1e-5);

        // log(exp(x)) should be x
        let log_result = exp_result.log().unwrap();
        log_result.eval();
        let log_vals = log_result.to_vec::<f32>().unwrap();
        assert!((log_vals[0] - 0.0).abs() < 1e-5);
        assert!((log_vals[1] - 1.0).abs() < 1e-5);
        assert!((log_vals[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_array_sin_cos() {
        let a = Array::from_slice(&[0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI], &[3]).unwrap();

        let sin_result = a.sin().unwrap();
        sin_result.eval();
        let sin_vals = sin_result.to_vec::<f32>().unwrap();
        assert!((sin_vals[0] - 0.0).abs() < 1e-5);  // sin(0) = 0
        assert!((sin_vals[1] - 1.0).abs() < 1e-5);  // sin(pi/2) = 1
        assert!(sin_vals[2].abs() < 1e-5);          // sin(pi) ≈ 0

        let cos_result = a.cos().unwrap();
        cos_result.eval();
        let cos_vals = cos_result.to_vec::<f32>().unwrap();
        assert!((cos_vals[0] - 1.0).abs() < 1e-5);  // cos(0) = 1
        assert!(cos_vals[1].abs() < 1e-5);          // cos(pi/2) ≈ 0
        assert!((cos_vals[2] + 1.0).abs() < 1e-5);  // cos(pi) = -1
    }

    #[test]
    fn test_array_max_min() {
        let a = Array::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], &[8]).unwrap();

        let max_result = a.max_all(false).unwrap();
        max_result.eval();
        assert_eq!(max_result.to_vec::<f32>().unwrap(), vec![9.0]);

        let min_result = a.min_all(false).unwrap();
        min_result.eval();
        assert_eq!(min_result.to_vec::<f32>().unwrap(), vec![1.0]);
    }

    #[test]
    fn test_array_sum_axis() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Sum along axis 0
        let sum0 = a.sum_axes(&[0], false).unwrap();
        sum0.eval();
        assert_eq!(sum0.to_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);

        // Sum along axis 1
        let sum1 = a.sum_axes(&[1], false).unwrap();
        sum1.eval();
        assert_eq!(sum1.to_vec::<f32>().unwrap(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_array_mean_all() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Mean of all elements: (1+2+3+4+5+6)/6 = 3.5
        let mean = a.mean_all(false).unwrap();
        mean.eval();
        assert_eq!(mean.to_vec::<f32>().unwrap(), vec![3.5]);
    }

    #[test]
    fn test_array_keepdims() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Sum with keepdims=true
        let sum_keep = a.sum_axes(&[1], true).unwrap();
        sum_keep.eval();
        assert_eq!(sum_keep.shape(), vec![2, 1]);
        assert_eq!(sum_keep.to_vec::<f32>().unwrap(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_array_squeeze() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3, 1]).unwrap();
        let squeezed = a.squeeze().unwrap();
        squeezed.eval();
        assert_eq!(squeezed.shape(), vec![3]);
    }

    #[test]
    fn test_array_expand_dims() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();

        let expanded0 = a.expand_dims(0).unwrap();
        expanded0.eval();
        assert_eq!(expanded0.shape(), vec![1, 3]);

        let expanded1 = a.expand_dims(1).unwrap();
        expanded1.eval();
        assert_eq!(expanded1.shape(), vec![3, 1]);
    }

    #[test]
    fn test_array_flatten() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let flat = a.flatten().unwrap();
        flat.eval();
        assert_eq!(flat.shape(), vec![6]);
        assert_eq!(flat.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_array_arange() {
        let a = Array::arange::<f32>(0.0, 5.0, 1.0).unwrap();
        a.eval();
        assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let b = Array::arange::<f32>(0.0, 1.0, 0.25).unwrap();
        b.eval();
        assert_eq!(b.to_vec::<f32>().unwrap(), vec![0.0, 0.25, 0.5, 0.75]);
    }

    #[test]
    fn test_array_linspace() {
        let a = Array::linspace::<f32>(0.0, 1.0, 5).unwrap();
        a.eval();
        let vals = a.to_vec::<f32>().unwrap();
        assert_eq!(vals.len(), 5);
        assert!((vals[0] - 0.0).abs() < 1e-5);
        assert!((vals[2] - 0.5).abs() < 1e-5);
        assert!((vals[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_array_eye() {
        let eye = Array::eye::<f32>(3, None, 0).unwrap();
        eye.eval();
        let vals = eye.to_vec::<f32>().unwrap();
        assert_eq!(vals, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_array_int_types() {
        // Test i32
        let a = Array::from_slice(&[1i32, 2, 3, 4], &[4]).unwrap();
        a.eval();
        assert_eq!(a.to_vec::<i32>().unwrap(), vec![1, 2, 3, 4]);

        // Test i64
        let b = Array::from_slice(&[100i64, 200, 300], &[3]).unwrap();
        b.eval();
        assert_eq!(b.to_vec::<i64>().unwrap(), vec![100, 200, 300]);
    }

    #[test]
    fn test_array_bool() {
        let a = Array::from_slice(&[true, false, true, false], &[4]).unwrap();
        a.eval();
        assert_eq!(a.to_vec::<bool>().unwrap(), vec![true, false, true, false]);
    }

    #[test]
    fn test_matmul_batch() {
        // Batch matrix multiplication (2, 2, 2) @ (2, 2, 2)
        let a = Array::from_slice(&[
            1.0f32, 2.0, 3.0, 4.0,  // First matrix
            5.0, 6.0, 7.0, 8.0,     // Second matrix
        ], &[2, 2, 2]).unwrap();

        let b = Array::from_slice(&[
            1.0f32, 0.0, 0.0, 1.0,  // Identity for first
            2.0, 0.0, 0.0, 2.0,     // 2*Identity for second
        ], &[2, 2, 2]).unwrap();

        let c = a.matmul(&b).unwrap();
        c.eval();
        assert_eq!(c.shape(), vec![2, 2, 2]);

        let vals = c.to_vec::<f32>().unwrap();
        // First batch: [[1,2],[3,4]] @ I = [[1,2],[3,4]]
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 2.0);
        // Second batch: [[5,6],[7,8]] @ 2I = [[10,12],[14,16]]
        assert_eq!(vals[4], 10.0);
        assert_eq!(vals[5], 12.0);
    }

    #[test]
    fn test_array_contiguous() {
        // Test that transposed arrays work correctly
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t = a.transpose().unwrap();
        t.eval();

        // Should be able to read values after transpose
        let vals = t.to_vec::<f32>().unwrap();
        assert_eq!(vals.len(), 6);
    }

    // ============================================================================
    // GPU Tests (Apple Silicon)
    // ============================================================================

    #[test]
    fn test_gpu_device_available() {
        use device::Device;

        // Check that GPU device can be created on Apple Silicon
        let _gpu = Device::gpu();
        let _cpu = Device::cpu();

        // Default device should be available
        let default = Device::default_device();
        assert!(default.is_gpu() || default.is_cpu());
    }

    #[test]
    fn test_gpu_array_creation() {
        use device::Device;
        use stream::Stream;

        // Create array on GPU
        let gpu = Device::gpu();
        let _stream = Stream::new(&gpu);

        let a = Array::zeros::<f32>(&[1000, 1000]).unwrap();
        a.eval();

        // Verify shape
        assert_eq!(a.shape(), vec![1000, 1000]);
        assert_eq!(a.size(), 1_000_000);
    }

    #[test]
    fn test_gpu_matmul_performance() {
        // Test larger matrix multiplication on GPU
        let n = 256;

        let a = Array::ones::<f32>(&[n, n]).unwrap();
        let b = Array::ones::<f32>(&[n, n]).unwrap();

        let c = a.matmul(&b).unwrap();
        c.eval();

        // Result should be n * 1.0 = n for all elements
        let vals = c.to_vec::<f32>().unwrap();
        assert_eq!(vals.len(), (n * n) as usize);
        assert!((vals[0] - n as f32).abs() < 1e-3);
        assert!((vals[(n * n - 1) as usize] - n as f32).abs() < 1e-3);
    }

    #[test]
    fn test_gpu_elementwise_ops() {
        // Test various elementwise operations on larger arrays
        let n = 10000;

        let a = Array::ones::<f32>(&[n]).unwrap();
        let b = Array::ones::<f32>(&[n]).unwrap();

        // Addition
        let c = &a + &b;
        c.eval();
        let vals = c.to_vec::<f32>().unwrap();
        assert!(vals.iter().all(|&x| (x - 2.0).abs() < 1e-6));

        // Multiplication
        let d = &a * &b;
        d.eval();
        let vals = d.to_vec::<f32>().unwrap();
        assert!(vals.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_gpu_reduction_large() {
        // Test reduction on larger array
        let n = 100000;

        let a = Array::ones::<f32>(&[n]).unwrap();
        let sum = a.sum_all(false).unwrap();
        sum.eval();

        let val = sum.to_vec::<f32>().unwrap()[0];
        assert!((val - n as f32).abs() < 1.0);  // Allow small floating point error
    }

    #[test]
    fn test_gpu_neural_network_forward() {
        // Simulate a small neural network forward pass on GPU
        let batch = 32;
        let in_features = 128;
        let hidden = 256;
        let out_features = 10;

        // Input
        let x = random::normal_with_params::<f32>(&[batch, in_features], 0.0, 1.0, None).unwrap();

        // Layer 1: Linear + ReLU
        let w1 = random::normal_with_params::<f32>(&[in_features, hidden], 0.0, 0.01, None).unwrap();
        let h1 = x.matmul(&w1).unwrap();
        let h1_relu = nn::relu(&h1).unwrap();

        // Layer 2: Linear + Softmax
        let w2 = random::normal_with_params::<f32>(&[hidden, out_features], 0.0, 0.01, None).unwrap();
        let out = h1_relu.matmul(&w2).unwrap();
        let probs = nn::softmax(&out, -1).unwrap();

        probs.eval();

        // Check output shape
        assert_eq!(probs.shape(), vec![batch, out_features]);

        // Check probabilities sum to 1
        let prob_sum = probs.sum_axes(&[1], false).unwrap();
        prob_sum.eval();
        let sums = prob_sum.to_vec::<f32>().unwrap();
        for s in sums {
            assert!((s - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_gpu_conv2d_forward() {
        // Test conv2d on GPU
        let batch = 4;
        let in_channels = 3;
        let out_channels = 16;
        let h = 32;
        let w = 32;
        let kernel = 3;

        // Input: (N, H, W, C) - MLX uses channels-last
        let input = random::normal_with_params::<f32>(
            &[batch, h, w, in_channels], 0.0, 1.0, None
        ).unwrap();

        // Weight: (C_out, kH, kW, C_in)
        let weight = random::normal_with_params::<f32>(
            &[out_channels, kernel, kernel, in_channels], 0.0, 0.01, None
        ).unwrap();

        let output = nn::conv2d(&input, &weight, (1, 1), (1, 1), (1, 1), 1).unwrap();
        output.eval();

        // Output shape: (N, H_out, W_out, C_out)
        // With padding=1 and stride=1, H_out = H, W_out = W
        assert_eq!(output.shape(), vec![batch, h, w, out_channels]);
    }

    #[test]
    fn test_gpu_attention_forward() {
        // Test attention mechanism on GPU
        let batch = 2;
        let seq_len = 64;
        let embed_dim = 128;
        let num_heads = 8;

        let query = random::normal_with_params::<f32>(
            &[batch, seq_len, embed_dim], 0.0, 0.1, None
        ).unwrap();
        let key = random::normal_with_params::<f32>(
            &[batch, seq_len, embed_dim], 0.0, 0.1, None
        ).unwrap();
        let value = random::normal_with_params::<f32>(
            &[batch, seq_len, embed_dim], 0.0, 0.1, None
        ).unwrap();

        let output = nn::multi_head_attention_simple(
            &query, &key, &value,
            num_heads,
            nn::AttentionMask::None,
            None,
        ).unwrap();

        output.eval();
        assert_eq!(output.shape(), vec![batch, seq_len, embed_dim]);
    }

    #[test]
    fn test_gpu_stream_synchronize() {
        use stream::Stream;
        use device::Device;

        let gpu = Device::gpu();
        let stream = Stream::new(&gpu);

        // Create and evaluate array
        let a = Array::ones::<f32>(&[1000, 1000]).unwrap();
        let b = Array::ones::<f32>(&[1000, 1000]).unwrap();
        let c = a.matmul(&b).unwrap();
        c.eval();

        // Synchronize stream
        stream.synchronize();

        // Should be able to read results after sync
        let shape = c.shape();
        assert_eq!(shape, vec![1000, 1000]);
    }

    #[test]
    fn test_gpu_memory_large_array() {
        // Test handling of larger arrays on GPU
        let size: i32 = 4096;

        let a = Array::zeros::<f32>(&[size, size]).unwrap();
        a.eval();

        // Verify we can access the data
        assert_eq!(a.size() as i32, size * size);
        assert_eq!(a.nbytes() as i32, size * size * 4);  // f32 = 4 bytes
    }

    #[test]
    fn test_gpu_chained_operations() {
        // Test lazy evaluation with chained operations on GPU
        let n = 1000;

        let a = Array::ones::<f32>(&[n, n]).unwrap();
        let b = Array::ones::<f32>(&[n, n]).unwrap();

        // Chain multiple operations (lazy)
        let c = &a + &b;           // 2
        let d = &c * &c;           // 4
        let e = d.sqrt().unwrap(); // 2
        let f = e.sum_all(false).unwrap();

        // Only evaluate at the end
        f.eval();

        let val = f.to_vec::<f32>().unwrap()[0];
        // Should be n * n * 2.0 = 2,000,000
        assert!((val - 2_000_000.0).abs() < 100.0);
    }
}
