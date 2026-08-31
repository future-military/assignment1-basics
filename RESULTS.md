# CS336 Assignment 1: Training Results

## 1. Overview

This project implements and trains a Transformer language model from
scratch for Stanford CS336 Assignment 1.

The completed pipeline includes:

- Byte-pair encoding tokenizer training
- Streaming corpus encoding
- Transformer language model
- Rotary positional embeddings
- RMSNorm and SwiGLU
- AdamW optimization
- Cosine learning-rate scheduling
- Gradient clipping and accumulation
- Resumable and preemption-safe checkpoints
- CSV experiment logging
- Text generation with temperature and top-k sampling
- Reproducible fixed-batch checkpoint evaluation
- Modal L4 GPU training and evaluation

The final model was selected through a controlled learning-rate sweep
under an identical model, dataset, batch size, and token budget.

## 2. Dataset and Tokenizer

The model was trained on TinyStoriesV2-GPT4.

| Item | Value |
|---|---:|
| Training tokens | 540,796,778 |
| Validation tokens | 5,461,210 |
| Token dtype | `uint16` |
| Vocabulary size | 10,000 |
| BPE merges | 9,743 |
| Special token | `<\|endoftext\|>` |

Prepared artifacts:

```text
artifacts/tinystories_10k_tokenizer.pkl
data/tinystories_10k_train_ids.npy
data/tinystories_10k_validation_ids.npy

## 3. Model Configuration

| Hyperparameter | Value |
|---|---:|
| Vocabulary size | 10,000 |
| Context length | 256 |
| Model dimension | 512 |
| Attention heads | 16 |
| Transformer layers | 4 |
| Feed-forward dimension | 1,344 |
| RoPE theta | 10,000 |
| Trainable parameters | Approximately 22.7M |
| Training dtype | `float32` |

The architecture uses pre-normalized Transformer blocks with RMSNorm,
causal multi-head self-attention, rotary positional embeddings, and
SwiGLU feed-forward layers.

## 4. Training Configuration

All learning-rate experiments used the same training budget.

| Setting | Value |
|---|---:|
| GPU | NVIDIA L4 |
| GPU memory | Approximately 22 GiB |
| Optimizer | AdamW |
| AdamW betas | `(0.9, 0.95)` |
| Weight decay | 0.1 |
| Maximum gradient norm | 1.0 |
| Micro-batch size | 32 |
| Gradient accumulation | 1 |
| Effective batch size | 32 |
| Tokens per optimizer step | 8,192 |
| Optimizer steps | 5,000 |
| Total training tokens | 40,960,000 |
| Warmup steps | 500 |
| Schedule | Linear warmup followed by cosine decay |
| Evaluation interval | 500 steps |
| Evaluation batches | 20 |
| Checkpoint interval | 500 steps |
| Throughput | Approximately 29K–30K tokens/second |
| Wall-clock time per run | Approximately 23–24 minutes |

Training data was copied from the persistent Modal Volume to local
ephemeral storage before training to avoid random-read overhead.

Checkpoints include:

- Model state
- Optimizer state
- Completed optimizer step
- Model configuration
- PyTorch RNG state
- CUDA RNG state
- NumPy RNG state
- Python RNG state

This allows training to resume safely after Modal GPU preemption.

## 5. Learning-Rate Sweep

The maximum learning rate was varied while keeping all other major
training conditions fixed. The minimum learning rate was set to one
tenth of the corresponding maximum.

Every final checkpoint was reevaluated with the same validation seed
and the same 100 validation batches.

Each fixed evaluation covered:

```text
32 × 256 × 100 = 819,200 validation tokens
```

### Fixed Evaluation Results

| Maximum LR | Validation loss | Perplexity | Approximate 95% loss interval |
|---:|---:|---:|---:|
| `3e-4` | 1.880718 | 6.558212 | 1.866502–1.894934 |
| `6e-4` | 1.729677 | 5.638834 | 1.715985–1.743370 |
| `1e-3` | 1.667185 | 5.297233 | 1.653721–1.680648 |
| `1.5e-3` | **1.637339** | **5.141468** | **1.624061–1.650616** |

The final `1.5e-3` model reduced validation perplexity by approximately:

- 21.6% relative to the `3e-4` baseline
- 8.8% relative to the `6e-4` model
- 2.9% relative to the `1e-3` model

The selected final checkpoint is:

```text
checkpoints/lr1p5e3_5000.pt
```

This is the best model within the tested learning-rate range. The
experiment does not establish that `1.5e-3` is the globally optimal
learning rate.

## 6. Best Checkpoint Observation

During training, the best checkpoint was selected using 20 randomly
sampled validation batches.

For the `6e-4` and `1e-3` experiments, the training-time evaluator
selected step 4,500. However, the later fixed 100-batch evaluation
showed that the final step-5,000 checkpoints were better.

| Experiment | Step 4,500 loss | Step 5,000 loss |
|---|---:|---:|
| `6e-4` | 1.742934 | **1.729677** |
| `1e-3` | 1.681664 | **1.667185** |

This demonstrates that a small validation sample can introduce enough
noise to select a slightly worse checkpoint. The best-checkpoint
mechanism worked correctly, but its measurement was noisy.

For final model selection, the reproducible 100-batch evaluator was
used instead.

## 7. Generation Results

Generation used a 10,000-token BPE tokenizer with top-k sampling.

Common settings:

```text
Prompt: "Once upon a time"
Maximum new tokens: 256
Top-k: 50
Seed: 42
```

### Temperature Comparison

| Temperature | Observation |
|---:|---|
| 0.5 | Most stable, but more repetitive and generic |
| 0.8 | Best balance between structure and diversity |
| 1.0 | More diverse, but entity and attribute consistency degraded |

A representative sample at temperature 0.8 began:

> Once upon a time, there was a little boy named Tim. Tim loved to play
> outside in the sun. One day, it was very cold. Tim was sad and went
> out to play.

The model successfully produced:

- Multi-sentence stories
- Named characters
- Dialogue
- Basic conflict and resolution
- Mostly correct English syntax
- Natural end-of-story behavior

Remaining qualitative errors included:

- Entity inconsistencies
- Contradictory attributes
- Objects unexpectedly speaking
- Weak long-range causal consistency
- Occasional missing spaces between generated tokens

For this checkpoint, a temperature between 0.6 and 0.8 with `top_k=50`
provided the best practical quality.

## 8. Reproduction

### Train the selected configuration

```bash
modal run --detach modal_train.py \
  --run-name lr1p5e3_5000 \
  --max-steps 5000 \
  --batch-size 32 \
  --gradient-accumulation-steps 1 \
  --max-learning-rate 0.0015 \
  --min-learning-rate 0.00015 \
  --warmup-steps 500 \
  --cosine-cycle-steps 5000 \
  --eval-interval 500 \
  --eval-batches 20 \
  --checkpoint-interval 500
```

### Run reproducible evaluation

```bash
modal run modal_evaluate.py \
  --checkpoint-names \
  "l4_baseline_5000.pt,lr6e4_5000.pt,lr1e3_5000.pt,lr1p5e3_5000.pt" \
  --output-name final_model_comparison.csv \
  --batch-size 32 \
  --eval-batches 100 \
  --seed 2026
```

### Generate text

```bash
uv run python -m cs336_basics.generate \
  --checkpoint-path checkpoints/lr1p5e3_5000.pt \
  --tokenizer-path artifacts/tinystories_10k_tokenizer.pkl \
  --prompt "Once upon a time" \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-k 50
```

## 9. Limitations

- The model is small compared with modern production language models.
- Training used only 40.96M sampled tokens from the larger corpus.
- The maximum context length is limited to 256 tokens.
- The learning-rate sweep does not cover every possible value.
- Final evaluation samples random validation windows rather than
  exhaustively evaluating the full validation corpus.
- Generation does not use a KV cache and recomputes the retained
  context for every generated token.
- Semantic consistency remains weaker than grammatical consistency.

## 10. Final Result

The project produced a complete, reproducible language-model pipeline
from tokenizer training through GPU training, checkpoint recovery,
evaluation, and generation.

The selected model achieved:

```text
Validation loss: 1.637339
Validation perplexity: 5.141468
Training tokens: 40,960,000
GPU: NVIDIA L4
```

