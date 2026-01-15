//! Basic Array Operations Example
//!
//! This example demonstrates fundamental array operations in mlx-rs.
//!
//! Run with: cargo run --example basic_arrays

use mlx_rs::{Array, ops};

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS Basic Array Operations ===\n");

    // -------------------------------------------------------------------------
    // Array Creation
    // -------------------------------------------------------------------------
    println!("1. Array Creation");
    println!("-----------------");

    // From slice
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    a.eval();
    println!("From slice (2x3):\n{:?}", a.to_vec::<f32>()?);

    // Zeros and ones
    let zeros = Array::zeros::<f32>(&[3, 3])?;
    let ones = Array::ones::<f32>(&[2, 4])?;
    zeros.eval();
    ones.eval();
    println!("Zeros (3x3): {:?}", zeros.to_vec::<f32>()?);
    println!("Ones (2x4): {:?}", ones.to_vec::<f32>()?);

    // Range and linspace
    let range = Array::arange::<f32>(0.0, 10.0, 2.0)?;
    let linspace = Array::linspace::<f32>(0.0, 1.0, 5)?;
    range.eval();
    linspace.eval();
    println!("Arange [0, 10, step=2]: {:?}", range.to_vec::<f32>()?);
    println!("Linspace [0, 1, n=5]: {:?}", linspace.to_vec::<f32>()?);

    // Identity matrix
    let eye = Array::eye::<f32>(3, None, 0)?;
    eye.eval();
    println!("Identity 3x3: {:?}", eye.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Arithmetic Operations
    // -------------------------------------------------------------------------
    println!("\n2. Arithmetic Operations");
    println!("------------------------");

    let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
    let y = Array::from_slice(&[4.0f32, 5.0, 6.0], &[3])?;

    let sum = &x + &y;
    let diff = &x - &y;
    let prod = &x * &y;
    let quot = &y / &x;

    sum.eval();
    diff.eval();
    prod.eval();
    quot.eval();

    println!("x = {:?}", x.to_vec::<f32>()?);
    println!("y = {:?}", y.to_vec::<f32>()?);
    println!("x + y = {:?}", sum.to_vec::<f32>()?);
    println!("x - y = {:?}", diff.to_vec::<f32>()?);
    println!("x * y = {:?}", prod.to_vec::<f32>()?);
    println!("y / x = {:?}", quot.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Broadcasting
    // -------------------------------------------------------------------------
    println!("\n3. Broadcasting");
    println!("---------------");

    let matrix = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let row = Array::from_slice(&[10.0f32, 20.0, 30.0], &[1, 3])?;

    let broadcasted = &matrix + &row;
    broadcasted.eval();

    println!("Matrix (2x3) + Row (1x3):");
    println!("Matrix: {:?}", matrix.to_vec::<f32>()?);
    println!("Row: {:?}", row.to_vec::<f32>()?);
    println!("Result: {:?}", broadcasted.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Matrix Multiplication
    // -------------------------------------------------------------------------
    println!("\n4. Matrix Multiplication");
    println!("------------------------");

    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2])?;

    let c = a.matmul(&b)?;
    c.eval();

    println!("A (2x2): {:?}", a.to_vec::<f32>()?);
    println!("B (2x2): {:?}", b.to_vec::<f32>()?);
    println!("A @ B: {:?}", c.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Reductions
    // -------------------------------------------------------------------------
    println!("\n5. Reductions");
    println!("-------------");

    let data = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let total_sum = data.sum_all(false)?;
    let total_mean = data.mean_all(false)?;
    let row_sum = data.sum_axes(&[1], false)?;

    total_sum.eval();
    total_mean.eval();
    row_sum.eval();

    println!("Data (2x3): {:?}", data.to_vec::<f32>()?);
    println!("Sum all: {:?}", total_sum.to_vec::<f32>()?);
    println!("Mean all: {:?}", total_mean.to_vec::<f32>()?);
    println!("Sum along axis 1 (row sums): {:?}", row_sum.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Shape Manipulation
    // -------------------------------------------------------------------------
    println!("\n6. Shape Manipulation");
    println!("---------------------");

    let original = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let reshaped = original.reshape(&[3, 2])?;
    let transposed = original.transpose()?;
    let flattened = original.flatten()?;

    reshaped.eval();
    transposed.eval();
    flattened.eval();

    println!("Original shape: {:?}", original.shape());
    println!("Reshaped to (3, 2): {:?}", reshaped.shape());
    println!("Transposed: {:?}", transposed.shape());
    println!("Flattened: {:?}", flattened.shape());

    // -------------------------------------------------------------------------
    // Mathematical Functions
    // -------------------------------------------------------------------------
    println!("\n7. Mathematical Functions");
    println!("-------------------------");

    let vals = Array::from_slice(&[0.0f32, 1.0, 2.0, 4.0], &[4])?;

    let exp_vals = vals.exp()?;
    let sqrt_vals = vals.sqrt()?;

    exp_vals.eval();
    sqrt_vals.eval();

    println!("Values: {:?}", vals.to_vec::<f32>()?);
    println!("exp(values): {:?}", exp_vals.to_vec::<f32>()?);
    println!("sqrt(values): {:?}", sqrt_vals.to_vec::<f32>()?);

    let angles = Array::from_slice(&[0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI], &[3])?;
    let sin_vals = angles.sin()?;
    let cos_vals = angles.cos()?;

    sin_vals.eval();
    cos_vals.eval();

    println!("Angles [0, pi/2, pi]: {:?}", angles.to_vec::<f32>()?);
    println!("sin(angles): {:?}", sin_vals.to_vec::<f32>()?);
    println!("cos(angles): {:?}", cos_vals.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Lazy Evaluation
    // -------------------------------------------------------------------------
    println!("\n8. Lazy Evaluation");
    println!("------------------");

    // Operations are lazy - they build a computation graph
    let a = Array::ones::<f32>(&[1000, 1000])?;
    let b = Array::ones::<f32>(&[1000, 1000])?;

    // These don't compute anything yet
    let c = &a + &b;
    let d = &c * &c;
    let e = d.sum_all(false)?;

    // Only this triggers computation
    e.eval();

    println!("Created 1000x1000 arrays and chained operations");
    println!("Result of sum((a + b)^2): {:?}", e.to_vec::<f32>()?);
    println!("(Expected: 1000 * 1000 * 4 = 4,000,000)");

    println!("\n=== Example Complete ===");
    Ok(())
}
