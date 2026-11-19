# Developing a Tiny Transformer Language Model

**PUSL 3205 – Large Language Models and Generative AI | Lab 3**

A GPT-style, decoder-only Transformer built **from scratch in PyTorch** and trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset to generate short children's stories, character by character.

```
Prompt:  Once upon a time
Output:  Once upon a time, there was a little girl named Lily. She loved to play
         outside in the sunshine and explore. One day, she saw a wise owl near
         the house...
```

## Model

| Setting | Value |
|---|---|
| Architecture | Decoder-only Transformer (pre-LayerNorm) |
| Parameters | 10.8 M |
| Layers / heads / embedding size | 6 / 6 / 384 |
| Context length | 256 characters |
| Tokenisation | Character-level (97 characters + `<EOS>`) |
| Training data | 20,000 stories (17.9 M characters) |
| Training | 5 epochs, ~16 min on an RTX 4050 Laptop GPU |

All components are implemented by hand: token and positional embeddings, masked multi-head self-attention, feed-forward layers, residual connections and layer normalisation.

## Results

| Model | Val loss | Perplexity | Bits per char |
|---|---|---|---|
| Uniform guessing | 4.585 | 98.0 | 6.62 |
| Unigram (character frequency) | 3.056 | 21.2 | 4.41 |
| **Tiny Transformer** | **0.637** | **1.89** | **0.92** |

Validation loss fell every epoch (0.848 → 0.730 → 0.680 → 0.651 → 0.637), with no overfitting.

### Experiments (2 epochs each)

| Configuration | Params | Val loss |
|---|---|---|
| 2 layers, 6 heads | 3.7 M | 0.827 |
| 4 layers, 6 heads | 7.2 M | 0.742 |
| 6 layers, 1 head | 10.8 M | 0.749 |
| 6 layers, 3 heads | 10.8 M | 0.728 |
| **6 layers, 6 heads** | **10.8 M** | **0.718** |

More layers and more attention heads both lowered the loss, with diminishing returns.

## Key findings
- Attention is what makes the model work: loss falls from **2.34** with one character of context to **0.59** with the full 256-character context.
- Replacing the manual attention with PyTorch's fused `scaled_dot_product_attention` stopped GPU memory overflowing into system RAM and made training **~7× faster**.
- The learning-rate schedule matters, not just the number of epochs: a 2-epoch run whose learning rate had fully decayed beat a 5-epoch run's epoch-2 checkpoint.
- Weaknesses: repetition, logic errors and plot drift, mainly due to the short context window and small model.

## Notebook contents
1. Load and explore TinyStories
2. Character-level tokenisation
3. Input/target sequences for next-token prediction
4. Transformer architecture, with sanity checks (output shape, initial loss, causal mask)
5. Initialisation: AdamW, warm-up + cosine learning-rate schedule
6. Training loop with mixed precision and gradient clipping
7. Loss curves
8. Perplexity vs baselines, and loss by context position
9. Story generation with temperature and top-k sampling
10. Experiments with layers, heads and epochs
11. Saving and reloading the model
12. Discussion of the mechanisms that improve performance

## Project structure
```
├── notebooks/
│   └── tiny_transformer.ipynb   # full lab
├── models/
│   └── tiny_transformer.pt      # trained weights + config + vocabulary + history
├── requirements.txt
└── README.md
```

## How to run
```bash
git clone https://github.com/RavinduBatawala/Developing-a-Tiny-Transformer-Language-Model.git
cd Developing-a-Tiny-Transformer-Language-Model
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)

# PyTorch with CUDA: get the exact command for your system from pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```
Then open `notebooks/tiny_transformer.ipynb` and run the cells in order. Training takes ~16 minutes on an RTX 4050; a CPU will work but is much slower. To skip training, load the saved model from `models/tiny_transformer.pt`.

## Tech stack
Python · PyTorch · Hugging Face Datasets · NumPy · Matplotlib

## Reference
Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759