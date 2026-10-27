<p align="center">
  <img src="assets/logo.svg" alt="retnet logo" width="112" height="112">
</p>

<h1 align="center">retnet</h1>

<p align="center"><strong>Byte-level RetNet experiments, from first principles.</strong></p>

<p align="center">
  A from-scratch, softmax-free retention layer for byte-level language models —
  small enough to read, complete enough to train.
</p>

<p align="center"><strong>English</strong> | <a href="README.zh-CN.md">简体中文</a></p>

---

## Why this exists

This is a personal research archive, not a product. It asks a narrow question:
inside a tiny byte-level language model, can a cheap, parallelisable mixer — a
learned exponential decay applied to a running sum of `K ⊙ V` — replace
attention or dilated convolution? And if it can, which stacking of retention and
causal-convolution layers works best?

The experiments were run in notebooks with Ray Tune, then stopped. There is no
paper, no benchmark against a baseline, and no saved result set. What remains is
the complete working trail: the layer library, the model definitions, the search
configurations, and the notebooks that produced them.

## Quick start

There is no package, CLI, test suite, or requirements file. The fastest entry
point is the reference notebook.

```bash
git clone https://github.com/ZisIsNotZis/retnet.git
cd retnet
pip install torch numpy matplotlib pandas tqdm "ray[tune]" hyperopt
jupyter lab ray3.ipynb
```

`ray3.ipynb` contains the clearest complete path: retention block → byte-level
model → a self-contained `train()` loop (Adam + OneCycleLR, bf16 autocast) →
three Ray Tune ASHA searches. Open it on a CUDA machine and run top to bottom.

**Data is not included.** Every model reads a raw byte corpus through
`nn.Dat`, which memory-maps the file and writes a `path.map` vocabulary sidecar:

```python
from nn import Dat

dat = Dat("py.txt", 32, 256)  # (file, epochs, batch, context length)
```

Point the placeholder at your own byte file. A simple place to start is a
lowercased source-code corpus; `code.sh` is the author's helper for building
one. The generated `.map`, `ray_results/`, and `log.txt` are git-ignored.

### The core idea

The retention block is assembled from ordinary layers. This is the whole
sketch, with the surrounding model left out:

```python
from nn import *

model = Seq(
    Embedding(68, 64),
    Ret(64),          # learned-decay retention
    Linear(64, 68),   # next-byte logits
)
```

## What it can do now

- **Build a retention mixer from scratch** — `nn.Decay` implements the
  exponential-decay cumulative sum in linear and log space; `nn.Ret` wires it
  into a multi-head key/value block with causal projections.
- **Compose models without a framework** — `Seq`, `Add`, `Mul`, `Dot`, `Res`,
  and `Stack` make the notebooks read like the architecture diagrams.
- **Train byte-level LMs** — memory-mapped byte data, bf16 autocast, Adam,
  OneCycleLR, and loss/accuracy reporting.
- **Search architectures cheaply** — Ray Tune ASHA experiments over widths,
  heads, kernel sizes, and decay ranges, with roughly 0.33–0.5 GPU per trial.
- **Explore retention + convolution hybrids** — plain retention stacks, causal
  convolution branches, and strided/dilated U-Net-style down/up-sampling.
- **Read the research trail** — 28 notebooks, 0 saved outputs, and a 420-line
  library that is the direct ancestor of [`cznn`](https://github.com/ZisIsNotZis/cznn).

**What is not here:** the `py.txt` corpora, checkpoints, loss curves, best
configs, tests, CI, or a requirements file. The notebooks are a record of what
was tried, not a reproducible benchmark.

## Architecture

```text
byte corpus
   │
   ├─ Dat                 memory-map + byte vocabulary sidecar
   │
   └─ Embedding(68, d)
        │
        └─ N × [ retention (Decay + causal projections) + causal convolution ]
             │
             └─ Linear(d, vocab) → next-byte logits
```

| Path | What it is |
| --- | --- |
| `nn.py` | The 420-line compositional layer library: `Ret`, `Decay`, `Dat`, `Seq`/`Add`/`Mul`/`Dot`/`Res`, parameter-count sizing helpers, and causal convolution utilities. |
| `nnn.py` | An earlier 142-line fork of the same ideas; no notebook imports it. Kept as part of the trail. |
| `ray3.ipynb` | **Start here.** The most complete notebook: `Ret`, `Model`, a self-contained `train()`, and three ASHA search stages. |
| `ray1.ipynb`, `ray2.ipynb` | The same lineage with a `Blk`/`Blks` block family (parallel retention + causal-conv branches), `get_blks`, and five successive ASHA searches. |
| `ray3cnn-Copy5.ipynb`, `Untitled9.ipynb` | The largest retention + convolution hybrid workbenches, with `Cov`/`Covd`/`Covu` and `torch.compile(torch.jit.trace(...))` training loops. |
| `Untitled-Copy1.ipynb` | Adjacent thread: byte-level CNN LM plus random architecture search and a generation sampler. |
| `code.sh` | Personal helper that collects source files into lowercase byte corpora. |
| `docs/project-status.md` | Evidence, boundaries, and the current classification of the archive. |

The retention idea is the one from *Retentive Network: A Successor to
Transformer for Large Language Modeling* (Sun et al., 2023), re-derived here in
the author's own layer style rather than copied from the paper.

## How to develop

- **Notebooks are the unit of work.** There is no build, no test command, and
  no package to install. Open a notebook and run it against your own corpus.
- **Keep the library local.** The notebooks import `from nn import *`; `nn.py`
  stays in the repository so those imports keep resolving. The canonical,
  published copy is [`cznn`](https://github.com/ZisIsNotZis/cznn).
- **Expect missing names in the scratch files.** Some notebooks depend on
  helpers defined in an earlier notebook or nowhere at all (for example,
  `ray3.ipynb` calls an undefined `size_select`). They are preserved as run,
  not normalized into a tutorial.
- **Do not commit generated material.** `*.map`, `ray_results/`, `log.txt`,
  `__pycache__/`, and notebook checkpoints are ignored. Corpora and weights do
  not belong in the repository.
- **Dependencies are listed in Quick start.** If a notebook needs more
  (`transformers`, `numba`, `llvmlite`, `greenlet`, `lark`, `smac`, `scipy`),
  install it explicitly; there is no lockfile to hide the difference.

## How to contribute

This is a dormant research archive, but corrections and reproducibility fixes
are welcome. Open an issue with the notebook and cell you are looking at, or a
focused pull request that keeps the archive faithful to what was run. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the short version.

## Roadmap

### Implemented

- Retention (`Decay`) in linear and log-space forms, with causal shifting.
- `Ret` blocks and compositional model building in `nn.py`.
- Byte-level `Dat` loader with a vocabulary sidecar.
- Self-contained training loops with bf16 autocast, Adam, and OneCycleLR.
- Ray Tune ASHA searches over model width, heads, kernels, and decay ranges.
- Retention + causal-convolution and U-Net-style hybrid experiments.
- Archive documentation and an AGPL-3.0-only license.

### Planned (not scheduled)

Nothing is actively being built. If the archive is revived, the highest-value
next steps are:

- A minimal runnable example with a small public byte corpus.
- A pinned environment recipe for the core `ray3` path.
- A write-up comparing retention, causal convolution, and the hybrids, with
  honest curves from a reproducible search.
- Saved best configurations and loss curves, so claims can be checked.

## Future vision

A small, trustworthy reference for retention-based byte-level language models:
one command to train a readable model on a public corpus, reproducible configs,
and honest curves sitting next to the full research trail. The layer library
already exists; the missing piece is the write-up and the reproducible path
around it.

## Relationship to `cznn`

`nn.py` in this directory **is** the library now published as
[`cznn`](https://github.com/ZisIsNotZis/cznn) (`cznn/__init__.py`). The published
package is canonical; this file is kept so the notebooks' unchanged imports
still work. A `diff` shows only the provenance header added by `cznn`.

## License

[AGPL-3.0-only](LICENSE). The retention idea itself comes from the RetNet paper
cited above; the implementation and the experiment record are the author's.
