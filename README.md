# java-gpt

A **from-scratch GPT implementation in pure Java** — a faithful port of [microgpt-c](../microgpt-c/microgpt.c), designed to learn how the GPT model works internally: forward pass, backward pass, attention, RMSNorm, Adam optimiser, and more.

---

## Table of Contents

- [About](#about)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Usage](#usage)
  - [Train](#train)
  - [Inference](#inference)
  - [Upload to Hugging Face](#upload-to-hugging-face)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Package Overview](#package-overview)

---

## About

`java-gpt` is an educational project that implements a minimal GPT (Generative Pre-trained Transformer) from scratch in Java. It is based on the `microgpt-c` project — a dependency-free C implementation — and demonstrates the same core ideas:

- Character-level tokenisation
- Token + positional embeddings
- Multi-head causal self-attention with a KV-cache
- RMSNorm layer normalisation
- Squared-ReLU MLP feed-forward layers
- Cross-entropy loss with softmax
- Manual backpropagation (no autograd framework)
- Adam optimiser with cosine learning-rate decay
- Model save / load (custom binary format)
- Upload to Hugging Face Hub

---

## Architecture

| Parameter         | Value         |
|-------------------|---------------|
| Embedding dim     | 32            |
| Attention heads   | 4             |
| Transformer layers| 1             |
| Context window    | 8 tokens      |
| MLP hidden dim    | 128 (4×32)    |
| Activation        | Squared ReLU  |
| Normalisation     | RMSNorm       |
| Optimiser         | Adam          |
| LR schedule       | Cosine decay  |
| Default LR        | 1e-2          |
| Default steps     | 5 000         |

> All hyper-parameters are defined in `GptConfig.java` and can be changed before recompiling.

---

## Project Structure

```
java-gpt/
├── pom.xml
├── instruction.md
├── README.md
└── src/main/java/com/krushna/gpt/
    ├── GptConfig.java                  ← All hyperparameters & constants
    ├── Main.java                       ← Entry point (train | infer | upload)
    │
    ├── util/
    │   ├── RandomUtils.java            ← xorshift64 PRNG, Gaussian sampling, weighted choice
    │   └── MathUtils.java              ← linearFwd/Bwd, rmsnormFwd/Bwd, softmax, sqReLU
    │
    ├── tokenizer/
    │   └── Tokenizer.java              ← Character-level tokeniser with BOS token
    │
    ├── training/
    │   ├── Dataset.java                ← Text file loader + Fisher-Yates shuffle
    │   ├── AdamOptimizer.java          ← Adam with cosine LR schedule
    │   └── GptTrainer.java             ← Training loop + text generation
    │
    ├── model/
    │   ├── ModelParams.java            ← Weights, gradients & Adam moment buffers
    │   ├── PosActivations.java         ← Saved activations per position (for backprop)
    │   └── GptModel.java               ← Forward pass, backward pass, KV-cache
    │
    └── io/
        ├── ModelSerializer.java        ← Binary save / load (custom .bin format)
        └── HuggingFaceUploader.java    ← Upload model + model card to HF Hub
```

---

## Prerequisites

| Tool    | Version   |
|---------|-----------|
| Java    | 17 or 21  |
| Maven   | 3.8+      |

---

## Build

```bash
# Clone the repo
git clone <repo-url>
cd java-gpt

# Compile and package a self-contained fat-jar
mvn package -q

# The runnable jar will be at:
# target/java-gpt-1.0.0-jar-with-dependencies.jar
```

---

## Usage

All modes are invoked through the single fat-jar. The first argument selects the mode.

### Train

Train the model on any plain-text file (one document/sentence per line):

```bash
java -jar target/java-gpt-1.0.0-jar-with-dependencies.jar \
  train <inputFile> [modelFile] [steps]
```

| Argument     | Default      | Description                              |
|--------------|-------------|------------------------------------------|
| `inputFile`  | `input.txt`  | Path to the training text file           |
| `modelFile`  | `model.bin`  | Path where the trained model is saved    |
| `steps`      | `5000`       | Number of gradient-descent steps         |

**Example:**

```bash
# Train on shakespeare.txt for 5000 steps, save to my_model.bin
java -jar target/java-gpt-1.0.0-jar-with-dependencies.jar \
  train shakespeare.txt my_model.bin 5000
```

**Sample output:**

```
num docs: 40000
vocab size: 67
num params: 34595
step    1 / 5000 | loss 4.2041
step    2 / 5000 | loss 4.1873
...
step 5000 / 5000 | loss 1.8342
Training complete.
Model saved to my_model.bin

--- inference ---
sample  1: helo
sample  2: worl
...
```

> **No arguments?** Running the jar with no arguments defaults to `train input.txt model.bin 5000`.

---

### Inference

Load a previously saved model and generate text samples:

```bash
java -jar target/java-gpt-1.0.0-jar-with-dependencies.jar \
  infer <modelFile> [numSamples] [temperature]
```

| Argument      | Default     | Description                                        |
|---------------|-------------|----------------------------------------------------|
| `modelFile`   | `model.bin` | Path to the saved binary model file                |
| `numSamples`  | `20`        | Number of sample sequences to generate             |
| `temperature` | `0.5`       | Sampling temperature (lower = more deterministic)  |

**Example:**

```bash
# Generate 10 samples with temperature 0.8
java -jar target/java-gpt-1.0.0-jar-with-dependencies.jar \
  infer my_model.bin 10 0.8
```

---

### Upload to Hugging Face

Upload the saved model binary and an auto-generated model card to the [Hugging Face Hub](https://huggingface.co):

```bash
java -jar target/java-gpt-1.0.0-jar-with-dependencies.jar \
  upload <modelFile> <hfToken> <owner> <repoName>
```

| Argument    | Description                                           |
|-------------|-------------------------------------------------------|
| `modelFile` | Path to the saved binary model file                   |
| `hfToken`   | Hugging Face write-access API token (`hf_xxxx`)       |
| `owner`     | Your Hugging Face username or organisation            |
| `repoName`  | Name of the target HF model repository               |

**Example:**

```bash
java -jar target/java-gpt-1.0.0-jar-with-dependencies.jar \
  upload my_model.bin hf_XXXXXXXXXX krushna java-gpt-shakespeare
```

The uploader will:
1. Create the repository if it does not already exist.
2. Upload `model.bin` to the repository.
3. Generate and upload a `README.md` model card.

> **Get your token:** [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — create a token with **write** access.

---

## Configuration

All compile-time constants are centralised in `GptConfig.java`:

```java
// Model architecture
public static final int N_EMBD     = 32;    // embedding dimension
public static final int N_HEAD     = 4;     // attention heads
public static final int N_LAYER    = 1;     // transformer layers
public static final int BLOCK_SIZE = 8;     // context window

// Training defaults
public static final float DEFAULT_LR        = 1e-2f;
public static final int   DEFAULT_NUM_STEPS = 5000;
public static final float DEFAULT_TEMPERATURE = 0.5f;

// Dataset limits
public static final int MAX_DOCS    = 85_000;
public static final int MAX_DOC_LEN = 512;
```

Edit `GptConfig.java` and rebuild to experiment with different model sizes or training settings.

---

## How It Works

### Forward Pass (per token position)

```
token_id → wte[token_id] + wpe[pos_id]   (embedding lookup)
         → RMSNorm
         → [for each layer]
              RMSNorm → Q, K, V projections
              → multi-head causal self-attention (with KV-cache)
              → O projection + residual
              → RMSNorm → MLP (fc1 → sqReLU → fc2) + residual
         → lm_head (linear)
         → logits over vocabulary
```

### Backward Pass

All gradients are computed manually:
- Softmax cross-entropy gradient
- Linear layer gradient (w.r.t. input and weights)
- RMSNorm gradient
- Squared-ReLU gradient
- Attention softmax gradient (via saved attention weights)
- KV-cache gradient accumulation across positions

### Optimiser

Adam with bias correction and cosine learning-rate decay:

```
lr_t = lr × 0.5 × (1 + cos(π × step / totalSteps))
m    = β₁ × m + (1 - β₁) × g
v    = β₂ × v + (1 - β₂) × g²
p   -= lr_t × (m / (1 - β₁^t)) / (√(v / (1 - β₂^t)) + ε)
```

### Model File Format (`.bin`)

```
[4 bytes]  magic: 0x474C4D42 ("GLMB")
[4 bytes]  version: 1
[4 bytes]  vocabSize
[4 bytes]  N_EMBD
[4 bytes]  N_HEAD
[4 bytes]  N_LAYER
[4 bytes]  BLOCK_SIZE
[4 bytes]  numUniqueChars
[n × 2 bytes]   unique vocabulary characters (UTF-16)
[numParams × 4 bytes]  all float32 weights in order
```

---

## Package Overview

| Package                    | Class                  | Responsibility                                      |
|---------------------------|------------------------|-----------------------------------------------------|
| `com.krushna.gpt`          | `GptConfig`            | Hyperparameter constants                            |
| `com.krushna.gpt`          | `Main`                 | CLI entry point – dispatches train / infer / upload |
| `com.krushna.gpt.util`     | `RandomUtils`          | xorshift64 PRNG, Gaussian sampling, weighted choice |
| `com.krushna.gpt.util`     | `MathUtils`            | Linear, RMSNorm, Softmax, sqReLU (fwd + bwd)        |
| `com.krushna.gpt.tokenizer`| `Tokenizer`            | Character vocabulary, encode/decode, BOS token      |
| `com.krushna.gpt.training` | `Dataset`              | Load text file, shuffle documents                   |
| `com.krushna.gpt.training` | `AdamOptimizer`        | Adam with cosine LR schedule                        |
| `com.krushna.gpt.training` | `GptTrainer`           | Training loop and inference generation              |
| `com.krushna.gpt.model`    | `ModelParams`          | All weight arrays, gradient buffers, Adam moments   |
| `com.krushna.gpt.model`    | `PosActivations`       | Cached activations per position for backprop        |
| `com.krushna.gpt.model`    | `GptModel`             | Forward pass, backward pass, KV-cache               |
| `com.krushna.gpt.io`       | `ModelSerializer`      | Binary save / load of model weights                 |
| `com.krushna.gpt.io`       | `HuggingFaceUploader`  | REST API upload to Hugging Face Hub                 |

---

## License

MIT
