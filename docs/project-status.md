# retnet project status

## Classification

Archived research scratchpad: byte-level language-model experiments around a
RetNet-style **retention** layer (softmax-free causal mixer = learned
exponential decay over a cumulative sum of `K ⊙ V`), built on the author's
compositional layer library and searched with Ray Tune. Notebook-only; no
package, no entrypoint.

## Status

Dormant / archive. Not maintained, never written up, never benchmarked against a
baseline. The notebooks date from late 2023 (Ray run directories are named
`2023-11-01`); the library files are dated 2024-12-03. The supporting library
`nn.py` graduated out of this directory and is now published separately as
`cznn`; this archive keeps its local copy so the notebooks still import.

## Evidence

- `nn.py` (420 lines) — `Ret` (retention), `Decay` (learned causal decayed
  cumsum, linear or log-space), `Dat` (byte memmap + `<127`-byte vocabulary
  sidecar), `Seq`/`Add`/`Mul`/`Dot`/`Res`, parameter-count sizing (`bisect`,
  `sized`, `MinModel`). `diff` against `cznn/cznn/__init__.py` shows only the
  6-line provenance docstring added by the published package.
- `ray3.ipynb` — the most complete single notebook: `Ret` (`Decay` +
  causal `SepConvLeft` + BN/SiLU), `Model`, a self-contained `train()` doing
  byte next-token prediction (`cross_entropy(model(x[:, :-1]).mT, x[:, 1:])`,
  Adam + OneCycleLR, 10 000 steps, loss/accuracy every 100), and three Ray Tune
  ASHA search stages over widths/heads/kernel-sizes/decay ranges.
- `ray1.ipynb`, `ray2.ipynb` — the same lineage with a `Blk`/`Blks` block family
  (parallel retention + causal-conv branches), parameter-budget sizing
  (`get_blks`), `train_blks`, five successive ASHA searches, and per-run
  `get_dataframe()` result tables.
- `ray3cnn-Copy5.ipynb` (933 code lines, 78 `Ret(...)` calls) and
  `Untitled9.ipynb` (95 `Ret(...)` calls) — the largest retention + strided/
  dilated causal-conv hybrid stacks (`Cov`/`Covd`/`Covu`, U-Net-ish
  down/up-sampling), with a `torch.compile(torch.jit.trace(...))` bf16 training
  loop.
- `Untitled-Copy3.ipynb`, `Untitled-Copy5.ipynb`, `Untitled-Copy6.ipynb` —
  earlier `retcore`/`decore`/`covcore` designs and a `param`-dict `Model`
  factory, including a Ray Tune search over `decay` variants.
- `Untitled-Copy1.ipynb` — adjacent thread: byte-level CNN LM with random
  architecture search (`randtrain`, `ModelC`, `multiprocess.Pool`) and a
  generation sampler.
- `nnn.py` (142 lines) — an earlier, unused fork of the library; no notebook
  imports it.
- `code.sh` — personal helper that `sudo find`s the filesystem to build
  lowercase token corpora for a byte LM.

## Boundaries

- **No data**: the `py.txt` byte corpora the models consume are not in the
  archive; `Dat` will only run against a file the user supplies. Absolute author
  paths were replaced with the relative placeholders `py.txt` / `ray_results/`.
- **No results**: none of the 28 notebooks has saved cell outputs; the only
  figure is one markdown image attachment in `Untitled4.ipynb`. No loss curves,
  best-config tables, or write-up survive. Claims about which architecture
  "won" cannot be verified from this archive.
- **No checkpoints**, no weights, no tests, no CI, no requirements file.
- **Not standalone**: several notebooks depend on names defined in another
  notebook or nowhere (e.g. `ray3.ipynb` calls an undefined `size_select`), and
  `dat` is often assumed to exist from a prior session.
- **Scratch-grade**: near-duplicate `Untitled*` / `Untitled-CopyN` files mixed
  with unrelated dead ends (greenlet/fiber scheduling, numba/llvmlite, fp8,
  asyncio, numpy sort/unique utilities).
- **License**: AGPL-3.0-only (see `LICENSE`). The retention idea itself is
  from *Retentive Network* (Sun et al., 2023), re-derived rather than copied.
- Canonical copy of the library is the published
  [`cznn`](https://github.com/ZisIsNotZis/cznn); the local `nn.py` is retained
  only so the notebooks' unchanged `from nn import *` imports keep working.
