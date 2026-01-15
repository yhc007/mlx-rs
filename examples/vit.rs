//! Vision Transformer (ViT) Example
//!
//! This example demonstrates the Vision Transformer model implementation
//! for image classification tasks.
//!
//! Run with: cargo run --example vit

use mlx_rs::nn::{ViTConfig, ViTModel, ViTWeights};
use mlx_rs::Array;

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS Vision Transformer (ViT) Example ===\n");

    // -------------------------------------------------------------------------
    // 1. ViT Configuration Presets
    // -------------------------------------------------------------------------
    println!("1. ViT Configuration Presets");
    println!("----------------------------");

    let base = ViTConfig::vit_base_patch16_224();
    println!("ViT-Base/16 (224x224):");
    println!("  image_size: {}", base.image_size);
    println!("  patch_size: {}", base.patch_size);
    println!("  hidden_size: {}", base.hidden_size);
    println!("  num_hidden_layers: {}", base.num_hidden_layers);
    println!("  num_attention_heads: {}", base.num_attention_heads);
    println!("  num_patches: {}", base.num_patches());

    let large = ViTConfig::vit_large_patch16_224();
    println!("\nViT-Large/16 (224x224):");
    println!("  hidden_size: {}", large.hidden_size);
    println!("  num_hidden_layers: {}", large.num_hidden_layers);
    println!("  num_attention_heads: {}", large.num_attention_heads);

    let small = ViTConfig::vit_small_patch16_224();
    println!("\nViT-Small/16 (224x224):");
    println!("  hidden_size: {}", small.hidden_size);
    println!("  num_hidden_layers: {}", small.num_hidden_layers);

    let tiny = ViTConfig::vit_tiny_patch16_224();
    println!("\nViT-Tiny/16 (224x224):");
    println!("  hidden_size: {}", tiny.hidden_size);
    println!("  num_hidden_layers: {}", tiny.num_hidden_layers);

    // -------------------------------------------------------------------------
    // 2. Custom Configuration
    // -------------------------------------------------------------------------
    println!("\n2. Custom Configuration (for demo)");
    println!("-----------------------------------");

    // Use a smaller config for demonstration
    let config = ViTConfig::new()
        .image_size(64)
        .patch_size(8)
        .num_channels(3)
        .hidden_size(128)
        .num_hidden_layers(4)
        .num_attention_heads(4)
        .intermediate_size(512)
        .num_classes(10);

    println!("Demo config:");
    println!("  image_size: {}x{}", config.image_size, config.image_size);
    println!("  patch_size: {}x{}", config.patch_size, config.patch_size);
    println!("  num_patches: {}", config.num_patches());
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    println!("  num_classes: {}", config.num_classes);

    // -------------------------------------------------------------------------
    // 3. Initialize Model and Weights
    // -------------------------------------------------------------------------
    println!("\n3. Initialize Model and Weights");
    println!("--------------------------------");

    let weights = ViTWeights::random(&config)?;
    let model = ViTModel::new(config.clone());
    println!("Model initialized with random weights");
    println!("Number of encoder layers: {}", weights.layers.len());

    // -------------------------------------------------------------------------
    // 4. Basic Forward Pass
    // -------------------------------------------------------------------------
    println!("\n4. Basic Forward Pass (Image Classification)");
    println!("---------------------------------------------");

    // Create a dummy image: (batch, height, width, channels) - NHWC format
    // MLX uses channels-last format
    let images = Array::zeros::<f32>(&[1, 64, 64, 3])?;
    println!("Input image shape: {:?} (batch, H, W, C)", images.shape());

    let logits = model.forward(&images, &weights)?;
    logits.eval();

    println!("Output logits shape: {:?} (batch, num_classes)", logits.shape());

    // -------------------------------------------------------------------------
    // 5. Feature Extraction (without classification head)
    // -------------------------------------------------------------------------
    println!("\n5. Feature Extraction");
    println!("---------------------");

    let (cls_token, sequence_output) = model.get_features(&images, &weights)?;
    cls_token.eval();
    sequence_output.eval();

    println!("CLS token shape: {:?} (batch, hidden_size)", cls_token.shape());
    println!("Sequence output shape: {:?} (batch, num_patches+1, hidden_size)", sequence_output.shape());
    println!("These features can be used for:");
    println!("  - Transfer learning");
    println!("  - Fine-tuning on downstream tasks");
    println!("  - Feature similarity comparisons");

    // -------------------------------------------------------------------------
    // 6. Patch Embeddings
    // -------------------------------------------------------------------------
    println!("\n6. Patch Embeddings");
    println!("--------------------");

    let patch_embeddings = model.get_patch_embeddings(&images, &weights)?;
    patch_embeddings.eval();

    let num_patches = config.num_patches();
    println!("Patch embeddings shape: {:?}", patch_embeddings.shape());
    println!("  - Includes CLS token at position 0");
    println!("  - {} image patches + 1 CLS token = {} total", num_patches, num_patches + 1);

    // -------------------------------------------------------------------------
    // 7. Batch Processing
    // -------------------------------------------------------------------------
    println!("\n7. Batch Processing");
    println!("--------------------");

    let batch_images = Array::zeros::<f32>(&[4, 64, 64, 3])?;
    println!("Batch input shape: {:?}", batch_images.shape());

    let batch_logits = model.forward(&batch_images, &weights)?;
    batch_logits.eval();

    println!("Batch output shape: {:?}", batch_logits.shape());

    // -------------------------------------------------------------------------
    // 8. Different Patch Sizes
    // -------------------------------------------------------------------------
    println!("\n8. Different Patch Sizes");
    println!("------------------------");

    println!("Patch size affects the number of patches and computation:");

    let config_p8 = ViTConfig::new().image_size(64).patch_size(8);
    println!("  Patch 8x8 on 64x64 image: {} patches", config_p8.num_patches());

    let config_p16 = ViTConfig::new().image_size(64).patch_size(16);
    println!("  Patch 16x16 on 64x64 image: {} patches", config_p16.num_patches());

    let config_p32 = ViTConfig::new().image_size(64).patch_size(32);
    println!("  Patch 32x32 on 64x64 image: {} patches", config_p32.num_patches());

    println!("\nSmaller patches = more patches = more computation but finer detail");
    println!("Larger patches = fewer patches = faster but coarser representation");

    // -------------------------------------------------------------------------
    // 9. Understanding ViT Architecture
    // -------------------------------------------------------------------------
    println!("\n9. ViT Architecture Overview");
    println!("----------------------------");

    println!("
    Input Image (H x W x C)
           |
           v
    +------------------+
    | Patch Embedding  |  Split into patches, linear projection
    +------------------+
           |
           v
    +------------------+
    | + CLS Token      |  Prepend learnable [CLS] token
    +------------------+
           |
           v
    +------------------+
    | + Position Emb   |  Add positional embeddings
    +------------------+
           |
           v
    +------------------+
    | Transformer      |  L layers of:
    | Encoder Blocks   |  - Multi-head Self-Attention
    |                  |  - MLP (Feed Forward)
    +------------------+
           |
           v
    +------------------+
    | Layer Norm       |  Final normalization
    +------------------+
           |
           v
    +------------------+
    | Classification   |  [CLS] token -> linear head
    | Head             |
    +------------------+
           |
           v
       Logits
    ");

    // -------------------------------------------------------------------------
    // 10. Example Use Cases
    // -------------------------------------------------------------------------
    println!("10. Example Use Cases");
    println!("---------------------");

    println!("\nViT can be used for:");
    println!("  - Image Classification (ImageNet, CIFAR, etc.)");
    println!("  - Transfer Learning (pretrain on large dataset, fine-tune)");
    println!("  - Feature Extraction (use as backbone)");
    println!("  - Vision-Language Models (combine with text encoders)");
    println!("  - Object Detection (with appropriate heads)");

    println!("\nTypical training pipeline:");
    println!("```rust");
    println!("// 1. Load pretrained weights");
    println!("let weights = serialize::load_safetensors(\"vit-base.safetensors\")?;");
    println!("");
    println!("// 2. Create model");
    println!("let config = ViTConfig::vit_base_patch16_224().num_classes(1000);");
    println!("let model = ViTModel::new(config);");
    println!("");
    println!("// 3. Forward pass on images (NHWC format)");
    println!("let images = load_images()?;  // (batch, 224, 224, 3)");
    println!("let logits = model.forward(&images, &weights)?;");
    println!("");
    println!("// 4. Get predictions");
    println!("let predictions = logits.argmax(-1, false)?;");
    println!("```");

    println!("\n=== Example Complete ===");
    Ok(())
}
