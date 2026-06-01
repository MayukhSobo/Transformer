
# Transformer: Attention Is All You Need

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Code Quality](https://github.com/MayukhSobo/Transformer/actions/workflows/code-quality.yml/badge.svg)](https://github.com/MayukhSobo/Transformer/actions/workflows/code-quality.yml)

Educational PyTorch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762).

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/MayukhSobo/Transformer.git
cd Transformer
```

Run the setup script — interactively or by passing a mode directly:

| Mode | What it installs | Git hooks |
|------|-----------------|-----------|
| `run` | Core runtime only | No |
| `dev` | Core + dev tools + test suite | Yes |

```bash
./scripts/setup          # interactive TUI picker
./scripts/setup run      # non-interactive
./scripts/setup dev      # non-interactive
```

## Usage

### Training

```bash
python main.py --config config.toml
```

### Testing

> Requires `dev` mode.

```bash
python test_runner.py              # pytest (default)
python test_runner.py coverage     # pytest + coverage report → reports/
python test_runner.py parallel     # pytest-xdist (all CPUs)
python test_runner.py benchmark    # benchmark tests only
python test_runner.py unittest     # stdlib unittest
```

### Code quality

> Requires `dev` mode.

```bash
./scripts/run-sanity-check         # lint + format + type check
```

## Project Structure

```
Transformer/
├── arch/                    # Core transformer modules
│   ├── attentions/         # Self, multi-head, and cross-attention
│   ├── encoder/            # Encoder stack
│   ├── decoder/            # Decoder stack
│   ├── embedding.py
│   ├── positional_encoding.py
│   ├── feed_forward.py
│   └── residual_add_norm.py
├── tokenizer/              # SentencePiece and word-level tokenizers
├── tests/
├── scripts/                # setup, run-sanity-check
├── config.toml
├── model.py
├── train.py
├── dataset.py
└── main.py
```

## Configuration

```toml
[model]
hidden_size = 512
max_seq_len = 512
n_heads = 8
n_layers = 6
ff_hidden_size = 2048
dropout_pe = 0.1

[tokenizer]
kind = "sentencepiece"    # or "word"
algorithm = "bpe"         # or "unigram"

[training]
batch_size = 32
epochs = 10
learning_rate = 0.0005
```

## References

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

## License

MIT
