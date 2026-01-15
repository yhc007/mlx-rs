//! Serialization Example
//!
//! This example demonstrates loading and saving model weights in various formats.
//!
//! Run with: cargo run --example serialization

use mlx_rs::{Array, serialize};
use std::collections::HashMap;

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS Serialization Example ===\n");

    // -------------------------------------------------------------------------
    // Safetensors Format (Recommended for LLMs)
    // -------------------------------------------------------------------------
    println!("1. Safetensors Format");
    println!("---------------------");

    // Create some model weights
    let mut weights = HashMap::new();
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2])?,
    );
    weights.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?,
    );
    weights.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        Array::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2])?,
    );

    println!("Created {} tensors:", weights.len());
    for (name, arr) in &weights {
        arr.eval();
        println!("  {}: shape {:?}", name, arr.shape());
    }

    // Save to file
    let safetensors_path = std::env::temp_dir().join("mlx_example.safetensors");
    serialize::save_safetensors(&safetensors_path, &weights)?;
    println!("\nSaved to: {:?}", safetensors_path);

    // Get info without loading all data
    let info = serialize::get_safetensors_info(&safetensors_path)?;
    println!("\nTensor info:");
    for (name, shape, dtype) in &info {
        println!("  {}: {:?} ({})", name, shape, dtype);
    }

    // Load back
    let loaded = serialize::load_safetensors(&safetensors_path)?;
    println!("\nLoaded {} tensors", loaded.len());

    // Verify
    let embed = loaded.get("model.embed_tokens.weight").unwrap();
    embed.eval();
    println!("Embed shape: {:?}", embed.shape());
    println!("Embed data: {:?}", embed.to_vec::<f32>()?);

    // Clean up
    let _ = std::fs::remove_file(&safetensors_path);

    // -------------------------------------------------------------------------
    // In-Memory Safetensors
    // -------------------------------------------------------------------------
    println!("\n2. In-Memory Safetensors");
    println!("------------------------");

    // Convert to bytes (for network transfer, embedding in binary, etc.)
    let bytes = serialize::save_safetensors_to_bytes(&weights)?;
    println!("Serialized to {} bytes", bytes.len());

    // Load from bytes
    let from_bytes = serialize::load_safetensors_from_bytes(&bytes)?;
    println!("Loaded {} tensors from bytes", from_bytes.len());

    // -------------------------------------------------------------------------
    // NumPy NPY Format (Single Array)
    // -------------------------------------------------------------------------
    println!("\n3. NumPy NPY Format");
    println!("-------------------");

    let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?;
    array.eval();
    println!("Original array: shape {:?}", array.shape());
    println!("Data: {:?}", array.to_vec::<f32>()?);

    let npy_path = std::env::temp_dir().join("mlx_example.npy");
    serialize::save_npy(&npy_path, &array)?;
    println!("\nSaved to: {:?}", npy_path);

    let loaded_npy = serialize::load_npy(&npy_path)?;
    loaded_npy.eval();
    println!("Loaded: shape {:?}", loaded_npy.shape());

    let _ = std::fs::remove_file(&npy_path);

    // -------------------------------------------------------------------------
    // NumPy NPZ Format (Multiple Arrays)
    // -------------------------------------------------------------------------
    println!("\n4. NumPy NPZ Format");
    println!("-------------------");

    let mut npz_arrays = HashMap::new();
    npz_arrays.insert("weights".to_string(), Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?);
    npz_arrays.insert("bias".to_string(), Array::from_slice(&[0.1f32, 0.2], &[2])?);
    npz_arrays.insert("indices".to_string(), Array::from_slice(&[0i32, 1, 2, 3], &[4])?);

    println!("Arrays to save:");
    for (name, arr) in &npz_arrays {
        arr.eval();
        println!("  {}: shape {:?}, dtype {:?}", name, arr.shape(), arr.dtype());
    }

    let npz_path = std::env::temp_dir().join("mlx_example.npz");
    serialize::save_npz(&npz_path, &npz_arrays)?;
    println!("\nSaved to: {:?}", npz_path);

    let loaded_npz = serialize::load_npz(&npz_path)?;
    println!("Loaded {} arrays:", loaded_npz.len());
    for (name, arr) in &loaded_npz {
        arr.eval();
        println!("  {}: shape {:?}", name, arr.shape());
    }

    let _ = std::fs::remove_file(&npz_path);

    // -------------------------------------------------------------------------
    // Different Data Types
    // -------------------------------------------------------------------------
    println!("\n5. Different Data Types");
    println!("-----------------------");

    let mut typed_tensors = HashMap::new();
    typed_tensors.insert("float32".to_string(), Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?);
    typed_tensors.insert("int32".to_string(), Array::from_slice(&[1i32, 2, 3], &[3])?);
    typed_tensors.insert("bool".to_string(), Array::from_slice(&[true, false, true], &[3])?);

    let bytes = serialize::save_safetensors_to_bytes(&typed_tensors)?;
    let loaded = serialize::load_safetensors_from_bytes(&bytes)?;

    println!("Roundtrip different types:");
    for (name, arr) in &loaded {
        arr.eval();
        println!("  {}: dtype {:?}, shape {:?}", name, arr.dtype(), arr.shape());
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
