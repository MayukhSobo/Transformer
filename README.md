
# Transformer: Attention Is All You Need

[![Pylint](https://github.com/MayukhSobo/Transformer/actions/workflows/pylint.yml/badge.svg?branch=main&event=push)](https://github.com/MayukhSobo/Transformer/actions/workflows/pylint.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-390/)

Educational implementation of the Transformer architecture from the ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762) paper, built with PyTorch.

## 🚀 Features

- **Complete Encoder-Decoder Architecture** with cross-attention
- **Modular Design** - each component can be studied independently  
- **Multiple Tokenizers** - SentencePiece and word-level tokenization
- **WMT14 Dataset Integration** - German-English translation
- **Educational Focus** - well-documented code with comprehensive docstrings
- **Production Ready** - proper error handling, logging, and testing

## 🛠️ Installation

```bash
git clone https://github.com/MayukhSobo/Transformer.git
cd Transformer

# Using conda (recommended)
conda env create -f environment.yml
conda activate transformer

# Or using pip
pip install -r requirements.txt
```

## 📖 Usage

### Basic Model Creation

```python
from model import build_transformer
from config import Config

config = Config(config_file="config.toml")
transformer, dataset = build_transformer(config)

# Forward pass
output = transformer.forward(src_batch, tgt_batch, src_pad_mask, tgt_pad_mask)
```

### Training

```python
python main.py                    # Train with default config
python main.py --config custom.toml  # Train with custom config
```

### Testing

```python
python test_runner.py             # Run all tests
python test_runner.py pytest     # Run with pytest
python test_runner.py coverage   # Generate coverage report
```

## 📁 Project Structure

```
Transformer/
├── arch/                    # Core transformer modules
│   ├── attentions/         # Self, multi-head, and cross-attention
│   ├── encoder/            # Encoder components
│   ├── decoder/            # Decoder components  
│   ├── embedding.py        # Token embeddings
│   ├── positional_encoding.py
│   ├── feed_forward.py
│   └── residual_add_norm.py
├── tokenizer/              # Tokenization utilities
├── tests/                  # Test suite
├── data/                   # Dataset directory
├── config.toml             # Model configuration
├── model.py               # Model creation and orchestration
├── train.py               # Training implementation
├── dataset.py             # Dataset loading and preprocessing
└── main.py                # CLI entry point
```

## ⚙️ Configuration

Default model configuration (~101 million parameters, using distinct embeddings):

```toml
[model]
vocab_size = 37000
hidden_size = 512
max_seq_len = 512
n_heads = 8
n_layers = 6
ff_hidden_size = 2048
dropout_pe = 0.1

[tokenizer]
kind = "sentencepiece"    # or "word"
algorithm = "bpe"         # or "unigram"
vocab_size = 32000

[training]
batch_size = 32
epochs = 10
learning_rate = 0.0005

[dataset]
path = "./data"
```

## 🎯 Architecture Highlights

- **Multi-Head Attention**: 8 heads with 64 dimensions each
- **Positional Encoding**: Sinusoidal encoding with non-learnable/fixed parameters
- **Feed-Forward**: Two-layer MLP (512 → 2048 → 512)
- **Residual Connections**: Post-norm architecture with LayerNorm
- **Cross-Attention**: Full encoder-decoder interaction

## 📊 Current Status

- ✅ **Complete Architecture**: Encoder, decoder, and cross-attention implemented
- ✅ **Tokenization**: SentencePiece and word-level tokenizers
- ✅ **Dataset Integration**: WMT14 German-English with streaming support
- ⚠️ **Training Pipeline**: Forward pass implemented, optimization in progress
- ✅ **Testing**: Comprehensive test suite with 10.00/10 pylint score

## 🔧 Development

```bash
# Install git hooks (required — enforces formatting before every commit)
pre-commit install

# Run tests
python test_runner.py

# Run with coverage
python test_runner.py coverage

# Check code quality
pylint $(git ls-files '*.py')

# Format code
black .

# Update conda environment after changes to environment.yml
conda env update -f environment.yml --prune
```

## 📚 References

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Implementation guide

## 📄 License

MIT License - Free to use for educational purposes.

---

**Educational transformer implementation with complete encoder-decoder architecture and cross-attention, ready for sequence-to-sequence tasks.**
