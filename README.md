# mlx-rs

Rust bindings for [Apple MLX](https://github.com/ml-explore/mlx), a machine learning framework designed for Apple Silicon.

## Features

- **Array Operations**: Create, manipulate, and perform arithmetic on N-dimensional arrays
- **Automatic Differentiation**: Forward-mode (JVP) and reverse-mode (VJP) autodiff
- **Neural Network Layers**: Activations, normalization, attention, convolution, pooling, dropout
- **Optimizers**: SGD, Adam, AdamW, RMSprop with full configuration options
- **Llama Model**: Complete implementation with GQA support for Llama 2/3 architectures
- **Linear Algebra**: Matrix operations, decompositions, and solvers
- **GPU Acceleration**: Seamless execution on Apple Silicon GPUs

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mlx-rs = { git = "https://github.com/yhc007/mlx-rs" }
```

### Requirements

- macOS with Apple Silicon (M1/M2/M3)
- [MLX C API](https://github.com/ml-explore/mlx-c) installed
- Rust 1.70+

## Quick Start

```rust
use mlx_rs::{Array, nn, transforms};

// Create arrays
let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

// Matrix multiplication
let c = a.matmul(&b).unwrap();
c.eval();

// Neural network operations
let x = Array::from_slice(&[1.0f32, -1.0, 2.0, -2.0], &[2, 2]).unwrap();
let activated = nn::relu(&x).unwrap();

// Automatic differentiation
let grad_fn = |inputs: &[Array]| -> Vec<Array> {
    let x = &inputs[0];
    vec![x * x]  // f(x) = x^2
};

let x = Array::from_slice(&[3.0f32], &[1]).unwrap();
let grads = transforms::grad(grad_fn, &[x]).unwrap();
// grads[0] = 6.0 (derivative of x^2 at x=3)
```

## Core Modules

### Array Operations

```rust
use mlx_rs::Array;

// Creation
let zeros = Array::zeros::<f32>(&[3, 3]).unwrap();
let ones = Array::ones::<f32>(&[2, 4]).unwrap();
let range = Array::arange::<f32>(0.0, 10.0, 1.0).unwrap();

// Arithmetic (supports operator overloading)
let sum = &a + &b;
let product = &a * &b;
let matmul = a.matmul(&b).unwrap();

// Reductions
let total = a.sum_all(false).unwrap();
let mean = a.mean_all(false).unwrap();

// Shape manipulation
let reshaped = a.reshape(&[1, 4]).unwrap();
let transposed = a.transpose().unwrap();
```

### Neural Network Layers

```rust
use mlx_rs::nn;

// Activations
let y = nn::relu(&x).unwrap();
let y = nn::gelu(&x).unwrap();
let y = nn::softmax(&x, -1).unwrap();

// Normalization
let y = nn::layer_norm(&x, &weight, &bias, 1e-5).unwrap();
let y = nn::rms_norm(&x, &weight, 1e-6).unwrap();

// Attention
let attn = nn::scaled_dot_product_attention(
    &queries, &keys, &values,
    Some(scale),
    nn::AttentionMask::Causal,
    None,
).unwrap();

// Convolution
let y = nn::conv2d(&input, &weight, (1, 1), (0, 0), (1, 1), 1).unwrap();

// Dropout (training mode)
let y = nn::dropout(&x, 0.1, true).unwrap();
```

### Automatic Differentiation

```rust
use mlx_rs::transforms;

// Gradient computation
let f = |inputs: &[Array]| vec![&inputs[0] * &inputs[0]];
let grads = transforms::grad(f, &[x]).unwrap();

// Value and gradient together
let (values, grads) = transforms::value_and_grad(f, &[x]).unwrap();

// Vector-Jacobian product (reverse mode)
let (primals, vjps) = transforms::vjp(f, &[x], &[cotangent]).unwrap();

// Jacobian-vector product (forward mode)
let (primals, jvps) = transforms::jvp(f, &[x], &[tangent]).unwrap();
```

### Optimizers

```rust
use mlx_rs::nn::{SGD, Adam, AdamW, RMSprop, Optimizer};

// SGD with momentum
let mut optimizer = SGD::new(0.01)
    .momentum(0.9)
    .weight_decay(1e-4)
    .nesterov(true);

// Adam
let mut optimizer = Adam::new(0.001)
    .betas(0.9, 0.999)
    .eps(1e-8);

// Training step
let new_params = optimizer.step(&params, &grads).unwrap();
```

### Llama Model

```rust
use mlx_rs::nn::{LlamaConfig, LlamaModel, LlamaWeights};

// Use preset configuration
let config = LlamaConfig::llama3_8b();

// Or customize
let config = LlamaConfig::new()
    .vocab_size(32000)
    .hidden_size(4096)
    .num_hidden_layers(32)
    .num_attention_heads(32);

// Create model
let model = LlamaModel::new(config);

// Forward pass
let logits = model.forward(&input_ids, &weights).unwrap();

// Text generation
let output_ids = model.generate(&input_ids, &weights, 100, 0.8).unwrap();
```

## Supported Operations

### Activations
- ReLU, LeakyReLU, PReLU, ELU, SELU, CELU
- GELU (exact and approximate), SiLU/Swish, Mish
- Sigmoid, Tanh, Softmax, LogSoftmax
- Hardtanh, Hardsigmoid, Hardswish
- GLU, SwiGLU

### Normalization
- LayerNorm, BatchNorm, GroupNorm, InstanceNorm
- RMSNorm (used in Llama)

### Loss Functions
- MSE, L1, Huber (Smooth L1)
- Binary Cross-Entropy (with and without logits)
- Cross-Entropy, Hinge Loss
- Triplet Margin Loss

### Positional Encodings
- Sinusoidal (Transformer)
- Rotary (RoPE, used in Llama)
- Learned embeddings

### Linear Algebra
- Matrix inverse, solve, Cholesky
- QR decomposition, SVD
- Eigenvalues, norms, cross product

## Examples

Run examples with:

```bash
cargo run --example basic_arrays    # Array creation and manipulation
cargo run --example neural_network  # NN layers, activations, attention
cargo run --example autodiff        # Gradients, VJP, JVP
cargo run --example optimizer       # SGD, Adam, AdamW, RMSprop
cargo run --example llama           # Llama model architecture
```

## License

MIT License

## Acknowledgments

- [Apple MLX Team](https://github.com/ml-explore/mlx) for the MLX framework
- [mlx-c](https://github.com/ml-explore/mlx-c) for the C API bindings
