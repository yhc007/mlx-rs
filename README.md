# mlx-rs

Rust bindings for [Apple MLX](https://github.com/ml-explore/mlx), a machine learning framework designed for Apple Silicon.

## Features

- **Array Operations**: Create, manipulate, and perform arithmetic on N-dimensional arrays
- **Automatic Differentiation**: Forward-mode (JVP) and reverse-mode (VJP) autodiff
- **Neural Network Layers**: Activations, normalization, attention, convolution, pooling, dropout
- **Optimizers**: SGD, Adam, AdamW, RMSprop with full configuration options
- **Llama Model**: Complete implementation with GQA support for Llama 2/3 architectures
- **BERT Model**: Encoder-only transformer for embeddings, classification, NLU tasks
- **Vision Transformer (ViT)**: Image classification with patch-based attention
- **Whisper Model**: Speech recognition with encoder-decoder transformer
- **GPT-2 Model**: Decoder-only transformer for text generation
- **Serialization**: Load/save safetensors (HuggingFace), npy/npz (NumPy) formats
- **Learning Rate Schedulers**: StepLR, CosineAnnealing, WarmupCosine, OneCycleLR, and more
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

### BERT Model

```rust
use mlx_rs::nn::{BertConfig, BertModel, BertWeights};

// Use preset configuration
let config = BertConfig::bert_base_uncased();

// Or customize
let config = BertConfig::new()
    .hidden_size(768)
    .num_hidden_layers(12)
    .num_attention_heads(12);

// Create model
let model = BertModel::new(config);

// Forward pass - returns (last_hidden_state, pooled_output)
let (hidden, pooled) = model.forward(&input_ids, None, None, &weights).unwrap();

// Different pooling strategies
let cls_embedding = model.get_pooled_output(&input_ids, None, None, &weights).unwrap();
let mean_embedding = model.get_mean_pooled(&input_ids, None, None, &weights).unwrap();
```

### Vision Transformer (ViT)

```rust
use mlx_rs::nn::{ViTConfig, ViTModel, ViTWeights};

// Use preset configuration
let config = ViTConfig::vit_base_patch16_224();

// Or customize
let config = ViTConfig::new()
    .image_size(224)
    .patch_size(16)
    .hidden_size(768)
    .num_hidden_layers(12)
    .num_attention_heads(12)
    .num_classes(1000);

// Create model
let model = ViTModel::new(config);

// Forward pass - images in NHWC format (batch, height, width, channels)
let images = Array::zeros::<f32>(&[1, 224, 224, 3]).unwrap();
let logits = model.forward(&images, &weights).unwrap();

// Feature extraction (without classification head)
let (hidden_states, cls_token) = model.get_features(&images, &weights).unwrap();
```

### Whisper Model

```rust
use mlx_rs::nn::{WhisperConfig, WhisperModel, WhisperWeights};

// Use preset configuration
let config = WhisperConfig::whisper_base();

// Or customize
let config = WhisperConfig::new()
    .n_mels(80)
    .n_audio_state(512)
    .n_audio_layer(6)
    .n_text_state(512)
    .n_text_layer(6);

// Create model
let model = WhisperModel::new(config);

// Encode audio (mel spectrogram): (batch, n_mels, n_frames)
let mel = Array::zeros::<f32>(&[1, 80, 3000]).unwrap();
let audio_features = model.encode(&mel, &weights).unwrap();

// Decode with token IDs
let tokens = Array::from_slice(&[50258i32, 50259], &[1, 2]).unwrap();
let logits = model.decode(&tokens, &audio_features, &weights).unwrap();

// Or full forward pass
let logits = model.forward(&mel, &tokens, &weights).unwrap();
```

### GPT-2 Model

```rust
use mlx_rs::nn::{GPT2Config, GPT2Model, GPT2Weights};

// Use preset configuration
let config = GPT2Config::gpt2_small();

// Or customize
let config = GPT2Config::new()
    .vocab_size(50257)
    .n_positions(1024)
    .n_embd(768)
    .n_layer(12)
    .n_head(12);

// Create model
let model = GPT2Model::new(config);

// Forward pass
let logits = model.forward(&input_ids, &weights).unwrap();

// Text generation with temperature
let output_ids = model.generate(&input_ids, &weights, 50, 0.8).unwrap();
```

### Serialization

```rust
use mlx_rs::serialize;
use std::collections::HashMap;

// Load weights from safetensors (HuggingFace format)
let weights = serialize::load_safetensors("model.safetensors").unwrap();

// Save weights to safetensors
let mut tensors = HashMap::new();
tensors.insert("weight".to_string(), array);
serialize::save_safetensors("output.safetensors", &tensors).unwrap();

// NumPy format for Python interop
let array = serialize::load_npy("weights.npy").unwrap();
serialize::save_npy("output.npy", &array).unwrap();

// NPZ (multiple arrays)
let arrays = serialize::load_npz("weights.npz").unwrap();
serialize::save_npz("output.npz", &arrays).unwrap();
```

### Learning Rate Schedulers

```rust
use mlx_rs::scheduler::{WarmupCosine, CosineAnnealingLR, StepLR, OneCycleLR, LRScheduler};
use mlx_rs::nn::{Adam, Optimizer};

// WarmupCosine - most common for transformers
let mut scheduler = WarmupCosine::new(0.001, 1000, 10000)  // max_lr, warmup_steps, total_steps
    .min_lr(1e-6);

// CosineAnnealingLR
let mut scheduler = CosineAnnealingLR::new(0.1, 1000).min_lr(1e-6);

// StepLR - decay by gamma every step_size steps
let mut scheduler = StepLR::new(0.1, 30, 0.1);  // initial_lr, step_size, gamma

// OneCycleLR - super-convergence training
let mut scheduler = OneCycleLR::new(0.1, 10000)
    .pct_start(0.3)
    .div_factor(25.0);

// Training loop
let mut optimizer = Adam::new(0.001);
for step in 0..10000 {
    // ... forward, backward ...
    let lr = scheduler.step();
    optimizer.set_learning_rate(lr);
    let new_params = optimizer.step(&params, &grads).unwrap();
}
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

### Learning Rate Schedulers
- StepLR, MultiStepLR, ExponentialLR
- LinearLR, PolynomialLR, ConstantLR
- CosineAnnealingLR, CosineAnnealingWarmRestarts
- WarmupCosine, WarmupLinear (for transformers)
- OneCycleLR (super-convergence)

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
cargo run --example bert            # BERT model for embeddings
cargo run --example vit             # Vision Transformer for images
cargo run --example whisper         # Whisper speech recognition
cargo run --example gpt2            # GPT-2 text generation
cargo run --example serialization   # Load/save safetensors, npy, npz
cargo run --example scheduler       # Learning rate schedulers
```

## License

MIT License

## Acknowledgments

- [Apple MLX Team](https://github.com/ml-explore/mlx) for the MLX framework
- [mlx-c](https://github.com/ml-explore/mlx-c) for the C API bindings
