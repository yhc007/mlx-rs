//! CLIP Model Example
//!
//! This example demonstrates the CLIP (Contrastive Language-Image Pre-training)
//! model for multimodal vision-language tasks.
//!
//! Run with: cargo run --example clip

use mlx_rs::nn::{
    CLIPConfig, CLIPVisionConfig, CLIPTextConfig,
    CLIPModel, CLIPWeights,
    CLIPVisionEncoder, CLIPVisionWeights,
    CLIPTextEncoder, CLIPTextWeights,
};
use mlx_rs::Array;

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS CLIP Model Example ===\n");

    // -------------------------------------------------------------------------
    // 1. CLIP Configuration Presets
    // -------------------------------------------------------------------------
    println!("1. CLIP Configuration Presets");
    println!("------------------------------");

    let vit_b32 = CLIPConfig::clip_vit_base_patch32();
    println!("CLIP ViT-B/32 (OpenAI default):");
    println!("  Vision: {}x{} image, {}x{} patches",
             vit_b32.vision.image_size, vit_b32.vision.image_size,
             vit_b32.vision.patch_size, vit_b32.vision.patch_size);
    println!("  Vision hidden: {}", vit_b32.vision.hidden_size);
    println!("  Text hidden: {}", vit_b32.text.hidden_size);
    println!("  Projection dim: {}", vit_b32.projection_dim);

    let vit_b16 = CLIPConfig::clip_vit_base_patch16();
    println!("\nCLIP ViT-B/16:");
    println!("  Patches: {}x{} (more patches = finer details)",
             vit_b16.vision.patch_size, vit_b16.vision.patch_size);

    let vit_l14 = CLIPConfig::clip_vit_large_patch14();
    println!("\nCLIP ViT-L/14:");
    println!("  Vision hidden: {}", vit_l14.vision.hidden_size);
    println!("  Vision layers: {}", vit_l14.vision.num_hidden_layers);
    println!("  Projection dim: {}", vit_l14.projection_dim);

    let vit_l14_336 = CLIPConfig::clip_vit_large_patch14_336();
    println!("\nCLIP ViT-L/14@336px:");
    println!("  Image size: {}x{} (higher resolution)",
             vit_l14_336.vision.image_size, vit_l14_336.vision.image_size);

    // -------------------------------------------------------------------------
    // 2. Custom Configuration (for demo)
    // -------------------------------------------------------------------------
    println!("\n2. Custom Configuration (for demo)");
    println!("-----------------------------------");

    let vision_config = CLIPVisionConfig::new()
        .image_size(64)
        .patch_size(16)
        .hidden_size(64)
        .num_hidden_layers(2)
        .num_attention_heads(4)
        .intermediate_size(128);

    let text_config = CLIPTextConfig::new()
        .vocab_size(1000)
        .max_position_embeddings(32)
        .hidden_size(64)
        .num_hidden_layers(2)
        .num_attention_heads(4)
        .intermediate_size(128);

    let config = CLIPConfig::new()
        .vision(vision_config.clone())
        .text(text_config.clone())
        .projection_dim(32);

    println!("Demo config:");
    println!("  Vision: {}x{} images, {} patches",
             config.vision.image_size, config.vision.image_size,
             config.vision.num_patches());
    println!("  Text: vocab={}, max_len={}",
             config.text.vocab_size, config.text.max_position_embeddings);
    println!("  Projection dim: {}", config.projection_dim);

    // -------------------------------------------------------------------------
    // 3. Initialize Model and Weights
    // -------------------------------------------------------------------------
    println!("\n3. Initialize Model and Weights");
    println!("--------------------------------");

    let weights = CLIPWeights::random(&config)?;
    let model = CLIPModel::new(config.clone());
    println!("Model initialized with random weights");
    println!("Vision encoder layers: {}", weights.vision.layers.len());
    println!("Text encoder layers: {}", weights.text.layers.len());

    // -------------------------------------------------------------------------
    // 4. Image Encoding
    // -------------------------------------------------------------------------
    println!("\n4. Image Encoding");
    println!("-----------------");

    // Create dummy images: (batch, height, width, channels) in NHWC format
    let images = Array::zeros::<f32>(&[2, 64, 64, 3])?;
    println!("Input images shape: {:?} (batch, H, W, C)", images.shape());

    let image_embeds = model.encode_image(&images, &weights)?;
    image_embeds.eval();

    println!("Image embeddings shape: {:?}", image_embeds.shape());
    println!("  - L2 normalized embeddings in shared space");

    // -------------------------------------------------------------------------
    // 5. Text Encoding
    // -------------------------------------------------------------------------
    println!("\n5. Text Encoding");
    println!("----------------");

    // Create dummy tokens: (batch, seq_len)
    let tokens = Array::from_slice(&[
        1i32, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
    ], &[2, 8])?;
    println!("Input tokens shape: {:?} (batch, seq_len)", tokens.shape());

    let text_embeds = model.encode_text(&tokens, &weights)?;
    text_embeds.eval();

    println!("Text embeddings shape: {:?}", text_embeds.shape());
    println!("  - L2 normalized embeddings in shared space");

    // -------------------------------------------------------------------------
    // 6. Image-Text Similarity
    // -------------------------------------------------------------------------
    println!("\n6. Image-Text Similarity");
    println!("------------------------");

    // Multiple images and text prompts
    let images = Array::zeros::<f32>(&[3, 64, 64, 3])?;
    let tokens = Array::from_slice(&[
        1i32, 2, 3, 4, 5, 0, 0, 0,  // "a photo of a cat"
        6, 7, 8, 9, 10, 0, 0, 0,   // "a photo of a dog"
    ], &[2, 8])?;

    let logits = model.forward(&images, &tokens, &weights)?;
    logits.eval();

    println!("Images: 3, Text prompts: 2");
    println!("Similarity logits shape: {:?} (images, texts)", logits.shape());
    println!("  - Higher values = more similar");

    // Get probabilities
    let (img_to_text, text_to_img) = model.get_probs(&images, &tokens, &weights)?;
    img_to_text.eval();
    text_to_img.eval();

    println!("Image-to-text probs shape: {:?}", img_to_text.shape());
    println!("Text-to-image probs shape: {:?}", text_to_img.shape());

    // -------------------------------------------------------------------------
    // 7. Zero-Shot Classification
    // -------------------------------------------------------------------------
    println!("\n7. Zero-Shot Classification");
    println!("----------------------------");

    // Images to classify
    let images = Array::zeros::<f32>(&[4, 64, 64, 3])?;

    // Pre-computed text embeddings for class labels
    // In practice, encode prompts like "a photo of a {class}"
    let num_classes = 5;
    let proj_dim = config.projection_dim;

    // Simulate class embeddings (in practice, use encode_text)
    let class_embeds = Array::from_slice(
        &vec![0.1f32; (num_classes * proj_dim) as usize],
        &[num_classes, proj_dim],
    )?;

    let probs = model.classify(&images, &class_embeds, &weights)?;
    probs.eval();

    println!("Input images: 4");
    println!("Number of classes: {}", num_classes);
    println!("Classification probs shape: {:?}", probs.shape());

    // -------------------------------------------------------------------------
    // 8. CLIP Architecture Overview
    // -------------------------------------------------------------------------
    println!("\n8. CLIP Architecture Overview");
    println!("------------------------------");

    println!("
    Image Input [batch, H, W, C]          Text Input [batch, seq_len]
              |                                      |
              v                                      v
    +--------------------+               +--------------------+
    | Patch Embedding    |               | Token Embedding    |
    | (reshape + linear) |               +--------------------+
    +--------------------+                         |
              |                                    v
              v                          +--------------------+
    +--------------------+               | + Position Embed   |
    | + Class Token      |               +--------------------+
    +--------------------+                         |
              |                                    v
              v                          +--------------------+
    +--------------------+               | Transformer        |
    | + Position Embed   |               | (causal attention) |
    +--------------------+               | N layers           |
              |                          +--------------------+
              v                                    |
    +--------------------+                         v
    | Pre-LN             |               +--------------------+
    +--------------------+               | Final LN           |
              |                          +--------------------+
              v                                    |
    +--------------------+                         v
    | Transformer        |               +--------------------+
    | (bidirectional)    |               | EOT Token Pooling  |
    | N layers           |               +--------------------+
    +--------------------+                         |
              |                                    v
              v                          +--------------------+
    +--------------------+               | Text Projection    |
    | Post-LN            |               +--------------------+
    +--------------------+                         |
              |                                    v
              v                          Text Embedding
    +--------------------+               [batch, proj_dim]
    | CLS Token Pooling  |                         |
    +--------------------+                         |
              |                                    |
              v                                    |
    +--------------------+                         |
    | Visual Projection  |                         |
    +--------------------+                         |
              |                                    |
              v                                    v
    Image Embedding                     +--------------------+
    [batch, proj_dim]                   |   L2 Normalize     |
              |                         +--------------------+
              |                                    |
              +---------------+--------------------+
                              |
                              v
                    +--------------------+
                    | Cosine Similarity  |
                    | * exp(logit_scale) |
                    +--------------------+
                              |
                              v
                    Similarity Logits
                    [batch_images, batch_texts]
    ");

    // -------------------------------------------------------------------------
    // 9. Individual Encoders
    // -------------------------------------------------------------------------
    println!("9. Individual Encoders");
    println!("----------------------");

    // Vision encoder alone
    let vision_weights = CLIPVisionWeights::random(&vision_config)?;
    let vision_encoder = CLIPVisionEncoder::new(vision_config);

    let images = Array::zeros::<f32>(&[2, 64, 64, 3])?;
    let vision_output = vision_encoder.forward(&images, &vision_weights)?;
    vision_output.eval();

    println!("Vision encoder output shape: {:?} (before projection)", vision_output.shape());

    // Text encoder alone
    let text_weights = CLIPTextWeights::random(&text_config)?;
    let text_encoder = CLIPTextEncoder::new(text_config);

    let tokens = Array::from_slice(&[1i32, 2, 3, 4, 5, 6, 7, 8], &[2, 4])?;
    let text_output = text_encoder.forward(&tokens, &text_weights)?;
    text_output.eval();

    println!("Text encoder output shape: {:?} (before projection)", text_output.shape());

    // -------------------------------------------------------------------------
    // 10. Example Use Cases
    // -------------------------------------------------------------------------
    println!("\n10. Example Use Cases");
    println!("---------------------");

    println!("\nCLIP can be used for:");
    println!("  - Zero-shot image classification");
    println!("  - Image-text retrieval (search)");
    println!("  - Image captioning (with decoder)");
    println!("  - Visual question answering");
    println!("  - Object detection (with region proposals)");
    println!("  - Semantic image segmentation");

    println!("\nTypical zero-shot classification pipeline:");
    println!("```rust");
    println!("// 1. Load model and weights");
    println!("let config = CLIPConfig::clip_vit_base_patch32();");
    println!("let weights = serialize::load_safetensors(\"clip.safetensors\")?;");
    println!("let model = CLIPModel::new(config);");
    println!("");
    println!("// 2. Encode class labels");
    println!("let class_prompts = vec![");
    println!("    \"a photo of a cat\",");
    println!("    \"a photo of a dog\",");
    println!("    \"a photo of a bird\",");
    println!("];");
    println!("let class_tokens = tokenizer.encode_batch(&class_prompts)?;");
    println!("let class_embeds = model.encode_text(&class_tokens, &weights)?;");
    println!("");
    println!("// 3. Classify images");
    println!("let images = load_images(&[\"image1.jpg\", \"image2.jpg\"])?;");
    println!("let probs = model.classify(&images, &class_embeds, &weights)?;");
    println!("");
    println!("// 4. Get predictions");
    println!("let predictions = probs.argmax(-1)?;");
    println!("```");

    println!("\nTypical image-text retrieval pipeline:");
    println!("```rust");
    println!("// 1. Encode all images in database");
    println!("let all_images = load_image_database()?;");
    println!("let image_embeds = model.encode_image(&all_images, &weights)?;");
    println!("");
    println!("// 2. Encode search query");
    println!("let query = \"a sunset over the ocean\";");
    println!("let query_tokens = tokenizer.encode(query)?;");
    println!("let query_embed = model.encode_text(&query_tokens, &weights)?;");
    println!("");
    println!("// 3. Find most similar images");
    println!("let similarities = query_embed.matmul(&image_embeds.transpose()?)?;");
    println!("let top_k = similarities.topk(10)?;");
    println!("```");

    println!("\n=== Example Complete ===");
    Ok(())
}
