//! Serialization module for loading and saving model weights.
//!
//! Supports:
//! - **safetensors**: HuggingFace's efficient tensor format (recommended for LLMs)
//! - **npz**: NumPy's compressed archive format (Python MLX compatibility)
//!
//! # Example
//!
//! ```ignore
//! use mlx_rs::serialize;
//! use std::collections::HashMap;
//!
//! // Load weights from safetensors
//! let weights = serialize::load_safetensors("model.safetensors")?;
//!
//! // Save weights to safetensors
//! serialize::save_safetensors("output.safetensors", &weights)?;
//!
//! // Load from npz
//! let arrays = serialize::load_npz("weights.npz")?;
//! ```

use crate::array::Array;
use crate::error::{Error, Result};
use crate::dtype::DType;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter, Cursor};
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::Mmap;
use safetensors::tensor::{SafeTensors, TensorView};
use safetensors::serialize as st_serialize;

// ============================================================================
// Safetensors Support
// ============================================================================

/// Load tensors from a safetensors file.
///
/// Returns a HashMap mapping tensor names to Arrays.
///
/// # Arguments
///
/// * `path` - Path to the .safetensors file
///
/// # Example
///
/// ```ignore
/// let weights = serialize::load_safetensors("model.safetensors")?;
/// let embed = weights.get("model.embed_tokens.weight").unwrap();
/// ```
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Array>> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::Io(format!("Failed to open file: {}", e)))?;

    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io(format!("Failed to mmap file: {}", e)))?;

    load_safetensors_from_bytes(&mmap)
}

/// Load tensors from safetensors bytes.
///
/// Useful for loading from memory or embedded data.
pub fn load_safetensors_from_bytes(data: &[u8]) -> Result<HashMap<String, Array>> {
    let tensors = SafeTensors::deserialize(data)
        .map_err(|e| Error::Serialization(format!("Failed to parse safetensors: {}", e)))?;

    let mut result = HashMap::new();

    for (name, tensor) in tensors.tensors() {
        let array = tensor_view_to_array(&tensor)?;
        result.insert(name.to_string(), array);
    }

    Ok(result)
}

/// Convert safetensors TensorView to mlx Array.
fn tensor_view_to_array(tensor: &TensorView) -> Result<Array> {
    let shape: Vec<i32> = tensor.shape().iter().map(|&x| x as i32).collect();
    let data = tensor.data();

    match tensor.dtype() {
        safetensors::Dtype::F32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Array::from_slice(&floats, &shape)
        }
        safetensors::Dtype::F16 => {
            // Convert f16 to f32 for now (MLX supports f16 but Rust doesn't have native f16)
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            // Create as f32, then potentially convert to f16 if needed
            Array::from_slice(&floats, &shape)
        }
        safetensors::Dtype::BF16 => {
            // Convert bf16 to f32
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect();
            Array::from_slice(&floats, &shape)
        }
        safetensors::Dtype::F64 => {
            let floats: Vec<f64> = data
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .collect();
            // Convert to f32 for MLX compatibility
            let floats_f32: Vec<f32> = floats.iter().map(|&x| x as f32).collect();
            Array::from_slice(&floats_f32, &shape)
        }
        safetensors::Dtype::I32 => {
            let ints: Vec<i32> = data
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Array::from_slice(&ints, &shape)
        }
        safetensors::Dtype::I64 => {
            let ints: Vec<i64> = data
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
                .collect();
            // Convert to i32 for broader compatibility
            let ints_i32: Vec<i32> = ints.iter().map(|&x| x as i32).collect();
            Array::from_slice(&ints_i32, &shape)
        }
        safetensors::Dtype::U8 => {
            let bytes: Vec<u8> = data.to_vec();
            Array::from_slice(&bytes, &shape)
        }
        safetensors::Dtype::I8 => {
            let bytes: Vec<i8> = data.iter().map(|&b| b as i8).collect();
            // Convert to i32
            let ints: Vec<i32> = bytes.iter().map(|&x| x as i32).collect();
            Array::from_slice(&ints, &shape)
        }
        safetensors::Dtype::BOOL => {
            let bools: Vec<bool> = data.iter().map(|&b| b != 0).collect();
            Array::from_slice(&bools, &shape)
        }
        _ => Err(Error::Serialization(format!(
            "Unsupported dtype: {:?}",
            tensor.dtype()
        ))),
    }
}

/// Convert half-precision float (f16) bits to f32.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = 0u32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let exp_f32 = 127 - 15 - e;
            f32::from_bits((sign << 31) | (exp_f32 << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
        }
    } else {
        // Normal
        let exp_f32 = exp + 127 - 15;
        f32::from_bits((sign << 31) | (exp_f32 << 23) | (mant << 13))
    }
}

/// Convert bfloat16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    // BF16 is just the upper 16 bits of F32
    f32::from_bits((bits as u32) << 16)
}

/// Save tensors to a safetensors file.
///
/// # Arguments
///
/// * `path` - Output path for the .safetensors file
/// * `tensors` - HashMap of tensor names to Arrays
///
/// # Example
///
/// ```ignore
/// let mut weights = HashMap::new();
/// weights.insert("weight".to_string(), array);
/// serialize::save_safetensors("output.safetensors", &weights)?;
/// ```
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, Array>,
) -> Result<()> {
    let data = save_safetensors_to_bytes(tensors)?;

    let mut file = File::create(path.as_ref())
        .map_err(|e| Error::Io(format!("Failed to create file: {}", e)))?;

    file.write_all(&data)
        .map_err(|e| Error::Io(format!("Failed to write file: {}", e)))?;

    Ok(())
}

/// Save tensors to safetensors bytes.
pub fn save_safetensors_to_bytes(tensors: &HashMap<String, Array>) -> Result<Vec<u8>> {
    let mut tensor_views: Vec<(String, Vec<usize>, safetensors::Dtype, Vec<u8>)> = Vec::new();

    for (name, array) in tensors {
        array.eval();

        let shape: Vec<usize> = array.shape().iter().map(|&x| x as usize).collect();
        let (dtype, data) = array_to_bytes(array)?;

        tensor_views.push((name.clone(), shape, dtype, data));
    }

    // Build the tensor map for serialization
    let tensor_map: HashMap<String, TensorView> = tensor_views
        .iter()
        .map(|(name, shape, dtype, data)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();

    let refs: Vec<(&String, TensorView)> = tensor_map.iter().map(|(k, v)| (k, v.clone())).collect();

    st_serialize(refs, &None)
        .map_err(|e| Error::Serialization(format!("Failed to serialize safetensors: {}", e)))
}

/// Convert Array to bytes and dtype for safetensors.
fn array_to_bytes(array: &Array) -> Result<(safetensors::Dtype, Vec<u8>)> {
    let dtype = array.dtype();

    match dtype {
        DType::Float32 => {
            let data: Vec<f32> = array.to_vec()?;
            let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            Ok((safetensors::Dtype::F32, bytes))
        }
        DType::Float16 => {
            // Read as f32, convert to f16 bytes
            let data: Vec<f32> = array.to_vec()?;
            let bytes: Vec<u8> = data.iter().flat_map(|x| f32_to_half(*x).to_le_bytes()).collect();
            Ok((safetensors::Dtype::F16, bytes))
        }
        DType::BFloat16 => {
            let data: Vec<f32> = array.to_vec()?;
            let bytes: Vec<u8> = data.iter().flat_map(|x| f32_to_bf16(*x).to_le_bytes()).collect();
            Ok((safetensors::Dtype::BF16, bytes))
        }
        DType::Int32 => {
            let data: Vec<i32> = array.to_vec()?;
            let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            Ok((safetensors::Dtype::I32, bytes))
        }
        DType::Int64 => {
            let data: Vec<i64> = array.to_vec()?;
            let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            Ok((safetensors::Dtype::I64, bytes))
        }
        DType::UInt8 => {
            let data: Vec<u8> = array.to_vec()?;
            Ok((safetensors::Dtype::U8, data))
        }
        DType::Bool => {
            let data: Vec<bool> = array.to_vec()?;
            let bytes: Vec<u8> = data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();
            Ok((safetensors::Dtype::BOOL, bytes))
        }
        _ => Err(Error::Serialization(format!(
            "Unsupported dtype for safetensors: {:?}",
            dtype
        ))),
    }
}

/// Convert f32 to half-precision float (f16) bits.
fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf or NaN
        if mant == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7C00 | ((mant >> 13) as u16).max(1)
        }
    } else if exp > 142 {
        // Overflow to infinity
        sign | 0x7C00
    } else if exp < 103 {
        // Underflow to zero
        sign
    } else if exp < 113 {
        // Subnormal
        let mant = mant | 0x800000;
        let shift = 113 - exp;
        sign | ((mant >> (shift + 13)) as u16)
    } else {
        // Normal
        let exp_f16 = (exp - 112) as u16;
        sign | (exp_f16 << 10) | ((mant >> 13) as u16)
    }
}

/// Convert f32 to bfloat16 bits.
fn f32_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

// ============================================================================
// NPZ Support (NumPy format)
// ============================================================================

/// Load arrays from a .npz file (NumPy compressed archive).
///
/// # Arguments
///
/// * `path` - Path to the .npz file
///
/// # Example
///
/// ```ignore
/// let arrays = serialize::load_npz("weights.npz")?;
/// let weight = arrays.get("weight").unwrap();
/// ```
pub fn load_npz<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Array>> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::Io(format!("Failed to open file: {}", e)))?;

    let mut archive = zip::ZipArchive::new(BufReader::new(file))
        .map_err(|e| Error::Io(format!("Failed to read zip archive: {}", e)))?;

    let mut result = HashMap::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)
            .map_err(|e| Error::Io(format!("Failed to read archive entry: {}", e)))?;

        let name = file.name().trim_end_matches(".npy").to_string();

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| Error::Io(format!("Failed to read npy data: {}", e)))?;

        let array = load_npy_from_bytes(&data)?;
        result.insert(name, array);
    }

    Ok(result)
}

/// Load a single .npy file.
pub fn load_npy<P: AsRef<Path>>(path: P) -> Result<Array> {
    let mut file = File::open(path.as_ref())
        .map_err(|e| Error::Io(format!("Failed to open file: {}", e)))?;

    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| Error::Io(format!("Failed to read file: {}", e)))?;

    load_npy_from_bytes(&data)
}

/// Load array from .npy bytes.
fn load_npy_from_bytes(data: &[u8]) -> Result<Array> {
    let mut cursor = Cursor::new(data);

    // Check magic number
    let mut magic = [0u8; 6];
    cursor.read_exact(&mut magic)
        .map_err(|e| Error::Serialization(format!("Failed to read magic: {}", e)))?;

    if &magic != b"\x93NUMPY" {
        return Err(Error::Serialization("Invalid npy magic number".into()));
    }

    // Read version
    let major = cursor.read_u8()
        .map_err(|e| Error::Serialization(format!("Failed to read version: {}", e)))?;
    let _minor = cursor.read_u8()
        .map_err(|e| Error::Serialization(format!("Failed to read version: {}", e)))?;

    // Read header length
    let header_len = if major >= 2 {
        cursor.read_u32::<LittleEndian>()
            .map_err(|e| Error::Serialization(format!("Failed to read header len: {}", e)))? as usize
    } else {
        cursor.read_u16::<LittleEndian>()
            .map_err(|e| Error::Serialization(format!("Failed to read header len: {}", e)))? as usize
    };

    // Read header
    let mut header = vec![0u8; header_len];
    cursor.read_exact(&mut header)
        .map_err(|e| Error::Serialization(format!("Failed to read header: {}", e)))?;

    let header_str = String::from_utf8_lossy(&header);

    // Parse header (simple parsing, not full Python dict parser)
    let (dtype_str, fortran_order, shape) = parse_npy_header(&header_str)?;

    // Read data
    let data_start = cursor.position() as usize;
    let array_data = &data[data_start..];

    // Convert based on dtype
    npy_data_to_array(array_data, &dtype_str, &shape, fortran_order)
}

/// Parse NPY header string to extract dtype, fortran_order, and shape.
fn parse_npy_header(header: &str) -> Result<(String, bool, Vec<i32>)> {
    // Header format: {'descr': '<f4', 'fortran_order': False, 'shape': (10, 20), }

    // Extract descr - find the pattern 'descr': '...'
    let descr_pattern = "'descr':";
    let descr_start = header.find(descr_pattern).ok_or_else(||
        Error::Serialization("Missing descr in header".into()))?;

    // Find the value after 'descr':
    let after_descr = &header[descr_start + descr_pattern.len()..];
    // Skip whitespace and find opening quote
    let trimmed = after_descr.trim_start();

    let dtype_str = if trimmed.starts_with('\'') {
        // Extract quoted string
        let start = 1; // Skip opening quote
        let end = trimmed[1..].find('\'').unwrap_or(trimmed.len() - 1) + 1;
        trimmed[start..end].to_string()
    } else {
        return Err(Error::Serialization("Invalid descr format".into()));
    };

    // Extract fortran_order
    let fortran_order = header.contains("'fortran_order': True") ||
                        header.contains("'fortran_order':True");

    // Extract shape
    let shape_start = header.find("'shape':").ok_or_else(||
        Error::Serialization("Missing shape in header".into()))?;
    let shape_paren_start = header[shape_start..].find('(').unwrap() + shape_start;
    let shape_paren_end = header[shape_paren_start..].find(')').unwrap() + shape_paren_start;
    let shape_str = &header[shape_paren_start+1..shape_paren_end];

    let shape: Vec<i32> = if shape_str.trim().is_empty() {
        vec![] // Scalar
    } else {
        shape_str
            .split(',')
            .filter_map(|s| s.trim().parse::<i32>().ok())
            .collect()
    };

    Ok((dtype_str, fortran_order, shape))
}

/// Convert NPY binary data to Array based on dtype.
fn npy_data_to_array(data: &[u8], dtype: &str, shape: &[i32], _fortran_order: bool) -> Result<Array> {
    // dtype format: '<f4' (little-endian float32), '>f8' (big-endian float64), etc.
    let is_little_endian = dtype.starts_with('<') || dtype.starts_with('|');
    let type_char = dtype.chars().nth(1).unwrap_or('f');
    let type_size: usize = dtype[2..].parse().unwrap_or(4);

    match (type_char, type_size) {
        ('f', 4) => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| {
                    if is_little_endian {
                        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                    } else {
                        f32::from_be_bytes([b[0], b[1], b[2], b[3]])
                    }
                })
                .collect();
            Array::from_slice(&floats, shape)
        }
        ('f', 8) => {
            let floats: Vec<f64> = data
                .chunks_exact(8)
                .map(|b| {
                    if is_little_endian {
                        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                    } else {
                        f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                    }
                })
                .collect();
            // Convert to f32
            let floats_f32: Vec<f32> = floats.iter().map(|&x| x as f32).collect();
            Array::from_slice(&floats_f32, shape)
        }
        ('f', 2) => {
            // f16
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|b| {
                    let bits = if is_little_endian {
                        u16::from_le_bytes([b[0], b[1]])
                    } else {
                        u16::from_be_bytes([b[0], b[1]])
                    };
                    half_to_f32(bits)
                })
                .collect();
            Array::from_slice(&floats, shape)
        }
        ('i', 4) => {
            let ints: Vec<i32> = data
                .chunks_exact(4)
                .map(|b| {
                    if is_little_endian {
                        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
                    } else {
                        i32::from_be_bytes([b[0], b[1], b[2], b[3]])
                    }
                })
                .collect();
            Array::from_slice(&ints, shape)
        }
        ('i', 8) => {
            let ints: Vec<i64> = data
                .chunks_exact(8)
                .map(|b| {
                    if is_little_endian {
                        i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                    } else {
                        i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                    }
                })
                .collect();
            let ints_i32: Vec<i32> = ints.iter().map(|&x| x as i32).collect();
            Array::from_slice(&ints_i32, shape)
        }
        ('i', 2) => {
            let ints: Vec<i16> = data
                .chunks_exact(2)
                .map(|b| {
                    if is_little_endian {
                        i16::from_le_bytes([b[0], b[1]])
                    } else {
                        i16::from_be_bytes([b[0], b[1]])
                    }
                })
                .collect();
            let ints_i32: Vec<i32> = ints.iter().map(|&x| x as i32).collect();
            Array::from_slice(&ints_i32, shape)
        }
        ('i', 1) => {
            let ints: Vec<i32> = data.iter().map(|&b| b as i8 as i32).collect();
            Array::from_slice(&ints, shape)
        }
        ('u', 1) => {
            Array::from_slice(data, shape)
        }
        ('u', 4) => {
            let uints: Vec<u32> = data
                .chunks_exact(4)
                .map(|b| {
                    if is_little_endian {
                        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
                    } else {
                        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
                    }
                })
                .collect();
            let ints: Vec<i32> = uints.iter().map(|&x| x as i32).collect();
            Array::from_slice(&ints, shape)
        }
        ('b', 1) => {
            let bools: Vec<bool> = data.iter().map(|&b| b != 0).collect();
            Array::from_slice(&bools, shape)
        }
        _ => Err(Error::Serialization(format!(
            "Unsupported numpy dtype: {}",
            dtype
        ))),
    }
}

/// Save arrays to a .npz file.
///
/// # Arguments
///
/// * `path` - Output path for the .npz file
/// * `arrays` - HashMap of array names to Arrays
pub fn save_npz<P: AsRef<Path>>(path: P, arrays: &HashMap<String, Array>) -> Result<()> {
    let file = File::create(path.as_ref())
        .map_err(|e| Error::Io(format!("Failed to create file: {}", e)))?;

    let mut zip = zip::ZipWriter::new(BufWriter::new(file));
    let options = zip::write::FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);

    for (name, array) in arrays {
        let npy_data = save_npy_to_bytes(array)?;

        zip.start_file(format!("{}.npy", name), options.clone())
            .map_err(|e| Error::Io(format!("Failed to start zip entry: {}", e)))?;

        zip.write_all(&npy_data)
            .map_err(|e| Error::Io(format!("Failed to write zip entry: {}", e)))?;
    }

    zip.finish()
        .map_err(|e| Error::Io(format!("Failed to finish zip: {}", e)))?;

    Ok(())
}

/// Save a single array to a .npy file.
pub fn save_npy<P: AsRef<Path>>(path: P, array: &Array) -> Result<()> {
    let data = save_npy_to_bytes(array)?;

    let mut file = File::create(path.as_ref())
        .map_err(|e| Error::Io(format!("Failed to create file: {}", e)))?;

    file.write_all(&data)
        .map_err(|e| Error::Io(format!("Failed to write file: {}", e)))?;

    Ok(())
}

/// Save array to .npy bytes.
fn save_npy_to_bytes(array: &Array) -> Result<Vec<u8>> {
    array.eval();

    let mut buffer = Vec::new();

    // Magic number
    buffer.extend_from_slice(b"\x93NUMPY");

    // Version 1.0
    buffer.push(1);
    buffer.push(0);

    // Build header
    let dtype_str = match array.dtype() {
        DType::Float32 => "<f4",
        DType::Float64 => "<f8",
        DType::Float16 => "<f2",
        DType::BFloat16 => "<f2", // Stored as f16
        DType::Int32 => "<i4",
        DType::Int64 => "<i8",
        DType::Int16 => "<i2",
        DType::Int8 => "<i1",
        DType::UInt8 => "|u1",
        DType::UInt32 => "<u4",
        DType::Bool => "|b1",
        _ => return Err(Error::Serialization(format!(
            "Unsupported dtype for npy: {:?}",
            array.dtype()
        ))),
    };

    let shape = array.shape();
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
    };

    let header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype_str, shape_str
    );

    // Pad header to 64-byte alignment
    let header_len = header.len();
    let padding = 64 - ((10 + header_len) % 64);
    let padded_header = format!("{}{}\n", header, " ".repeat(padding - 1));

    // Write header length (2 bytes for v1.0)
    buffer.write_u16::<LittleEndian>(padded_header.len() as u16)
        .map_err(|e| Error::Io(format!("Failed to write header len: {}", e)))?;

    // Write header
    buffer.extend_from_slice(padded_header.as_bytes());

    // Write data
    let data_bytes = array_to_npy_bytes(array)?;
    buffer.extend_from_slice(&data_bytes);

    Ok(buffer)
}

/// Convert Array to bytes for npy format.
fn array_to_npy_bytes(array: &Array) -> Result<Vec<u8>> {
    match array.dtype() {
        DType::Float32 => {
            let data: Vec<f32> = array.to_vec()?;
            Ok(data.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        DType::Float64 => {
            let data: Vec<f64> = array.to_vec()?;
            Ok(data.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        DType::Int32 => {
            let data: Vec<i32> = array.to_vec()?;
            Ok(data.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        DType::Int64 => {
            let data: Vec<i64> = array.to_vec()?;
            Ok(data.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        DType::UInt8 => {
            let data: Vec<u8> = array.to_vec()?;
            Ok(data)
        }
        DType::Bool => {
            let data: Vec<bool> = array.to_vec()?;
            Ok(data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect())
        }
        _ => Err(Error::Serialization(format!(
            "Unsupported dtype for npy: {:?}",
            array.dtype()
        ))),
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get tensor names from a safetensors file without loading all data.
pub fn get_safetensors_info<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Vec<usize>, String)>> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::Io(format!("Failed to open file: {}", e)))?;

    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io(format!("Failed to mmap file: {}", e)))?;

    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| Error::Serialization(format!("Failed to parse safetensors: {}", e)))?;

    let info: Vec<(String, Vec<usize>, String)> = tensors
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            (
                name.to_string(),
                tensor.shape().to_vec(),
                format!("{:?}", tensor.dtype()),
            )
        })
        .collect();

    Ok(info)
}
