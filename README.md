# retnet — byte-level RetNet / retention experiments (scratch archive)

A personal research scratchpad: small byte-level language models whose sequence
mixer is a **RetNet-style retention layer** — a softmax-free causal
"attention" computed as a learned exponential decay applied to a running sum of
`K ⊙ V` — built on top of the author's compositional layer library `nn.py` and
hyper-parameter-searched with Ray Tune.

It exists because the author wanted to know whether a cheap, parallelisable,
recurrent-at-inference mixer (decayed cumulative sum) could replace attention /
dilated convolution inside a tiny byte-level LM, and which stacking of
retention + causal convolution layers wins. The problem being attacked is the
usual one: match the quality of attention on long-range byte dependencies at a
fraction of the cost, and find the layer topology that does it. No result was
ever written up.

## How to run

**There is no single entrypoint.** This is `.ipynb`-only research: 29 notebooks,
no package, no CLI, no tests, no requirements file.

1. Install (the core RetNet line only needs the first line):
   ```bash
   pip install torch numpy matplotlib pandas tqdm "ray[tune]" hyperopt
   ```
   Peripheral notebooks additionally use `transformers` (Untitled4),
   `numba` + `llvmlite` (numba.ipynb), `greenlet` + `lark` (Untitled9,
   play.ipynb), `smac` + `ConfigSpace` (Untitled2), `multiprocess` and
   `scipy` (Untitled4, Untitled-Copy1).
2. Open **`ray3.ipynb`** — the single most complete notebook (retention block +
   model + self-contained `train()` + three Ray Tune ASHA search stages).
   `ray1.ipynb` / `ray2.ipynb` are the same lineage with more search stages;
   `ray3cnn-Copy5.ipynb` and `Untitled9.ipynb` are the largest (messy)
   retention + CNN hybrid workbenches.
3. Run top-to-bottom in a CUDA environment. The training loop expects bf16
   autocast and runs 10 000–20 000 steps; notebooks were aimed at 0.5–0.33 GPU
   per Tune trial.
4. **Data is not included.** Every model reads a raw byte corpus through
   `nn.Dat(path, ...)`, which memory-maps the file and writes a `path.map`
   vocabulary sidecar (byte vocabulary < 127). Provide your own byte file and
   point the placeholder path at it:

   ```python
   dat = Dat("py.txt", 32, 256)   # (file, epochs, batch, context-length)
   ```

   The original author's absolute corpus paths (`/home/<user>/...`) were
   replaced with the relative placeholders `py.txt` and `ray_results/` so the
   notebooks carry no personal filesystem paths.

Nothing is checkpointed. The only disk output some notebooks produce is an
optional `log.txt`; everything else goes to matplotlib windows — so a completed
run leaves no artefact behind.

## What is actually here vs not

**Here**

- `nn.py` (420 lines) — the compositional layer library: `Seq/Add/Mul/Dot/Res`,
  `Ret` (retention), `Decay` (learned causal decayed cumsum), `Dat` (byte
  memmap loader), parameter-count sizing helpers.
- `nnn.py` (142 lines) — an earlier, unused fork of the same ideas; no notebook
  imports it.
- 29 notebooks, ≈7 057 non-empty code lines (7 713 raw), spanning the retention
  line and a lot of unrelated dead ends (greenlet/fiber scheduling, numba +
  llvmlite codegen, fp8, asyncio experiments, numpy sort/unique utilities,
  model-repr de-duplication).
- `code.sh` — a personal helper that `sudo find`s the whole filesystem to build
  lowercase token corpora for a language model.

**Not here**

- **No data / corpus.** The `py.txt` byte files the models train on are not in
  the archive.
- **No checkpoints** or model weights.
- **No results.** There are *no saved cell outputs anywhere* in the 29
  notebooks; the only figure is one markdown image attachment in
  `Untitled4.ipynb` (a bar chart of a weight histogram). No loss curves, no
  best-config tables, no write-up.
- **No tests, no CI, no requirements file, no license.**

**Honest assessment.** This is a pile of near-duplicate `Untitled*` and
`Untitled-CopyN` scratch notebooks, not a coherent project. The coherent thread
is real but short: a handful of notebooks (`ray1`, `ray2`, `ray3`,
`ray3cnn-Copy5`, `Untitled9`, `Untitled10`) revisit the same
`Embedding → [Ret + causal-comp conv] × N → Linear` idea with slightly different
building blocks, while the rest are unrelated experiments parked in the same
directory. Cells often reference utilities defined in an earlier notebook or
nowhere (e.g. `ray3.ipynb` calls an undefined `size_select`), so a notebook is
not guaranteed to run standalone even with data.

## Notebook map

| Notebook | Role |
| --- | --- |
| **`ray3.ipynb`** | **Most complete.** `Ret` (retention via `Decay` + causal `SepConvLeft`) → `Model` → self-contained `train()` (byte `Dat`, Adam + OneCycleLR, loss/accuracy reporting) + 3 Ray Tune ASHA searches. |
| `ray1.ipynb`, `ray2.ipynb` | Same lineage: `Ret`/`Cov` blocks combined into `Blk`/`Blks`, parameter-count sizing (`get_blks`), `train_blks`, 5 Tune stages with result tables and plots. |
| `ray3cnn-Copy5.ipynb` | Largest retention workbench (933 code lines, 78 `Ret(...)` uses): retention + strided/dilated conv U-Net-style stack (`Cov`/`Covd`/`Covu`), plain `torch.compile(torch.jit.trace(...))` training loop; heavily duplicated, expects a `dat` from elsewhere. |
| `Untitled9.ipynb` | Same model zoo as above plus greenlet/fiber scheduling experiments. |
| `Untitled10.ipynb` | The same `Ret` + conv hybrid at 1024 context. |
| `Untitled-Copy3.ipynb`, `Untitled-Copy5.ipynb`, `Untitled-Copy6.ipynb` | Earlier `retcore`/`decore`/`covcore` designs, a `Model` factory, and a Ray Tune search over `decay` variants. |
| `Untitled-Copy1.ipynb` | Separate thread: byte-level **CNN** LM + random architecture search (`randtrain`, `ModelC`), plus a sampler that prints generated text. |
| `ray3cnn.ipynb`, `ray3cnn-Copy1..4.ipynb` | CNN-only byte LMs (`SepConvLeft`/`Fullin`), model-string compression, and logging variants — conv research adjacent to the retention line. |
| `Untitled.ipynb`, `Untitled4.ipynb`, `Untitled5/6/7/8.ipynb`, `numba.ipynb`, `play.ipynb`, `fp8.ipynb`, `Untitled1/2/3/10/11` | Dead ends: numpy utility kernels, asyncio/generator experiments, ChatGLM-3-6B weight histograms, numba/llvmlite codegen, fp8, SMAC searcher glue, log parsing. |

The `decay` / `Ret` idea is the same everywhere: scale the key-value product by
`exp(λ · position)` with a learned per-channel λ, take a (log-)cumulative sum,
and normalise — the retention kernel from *Retentive Network: A Successor to
Transformer for Large Language Modeling* (Sun et al., 2023), re-derived here in
the author's own layer style rather than copied from the paper.

## Relationship to `cznn`

`nn.py` in this directory **is** the library that is now published as
[`cznn`](https://github.com/ZisIsNotZis/cznn) (`cznn/__init__.py`).

- **Canonical is `cznn`** — the published package. Its `cznn/__init__.py`
  is this file verbatim plus a 6-line provenance docstring; a `diff` shows only
  that header.
- The local `nn.py` is **kept here on purpose** so the notebooks'
  `from nn import *` imports still resolve. It is not a second maintained fork,
  and it is not deleted.
- The notebooks' imports are deliberately **not** rewritten to `import cznn`,
  so the archive stays faithful to what was actually run in 2023/2024. If you
  want the published library, install/use `cznn` instead.

## Status

Research / archive. Unmaintained, incomplete, not production code. Nothing here
was ever written up, released, or benchmarked against a baseline. Kept as a
record of the author's retention experiments and as the provenance of `cznn`.

## License

None. There is no `LICENSE` file; treat this as unpublished personal research
and assume all rights reserved by the author. Do not redistribute. Some cells
implement ideas from published papers (RetNet); those ideas remain the work of
their original authors.

## Provenance note

Copied read-only from the private archive `~/my/retnet`; nothing in the source
was modified. Absolute author filesystem paths were replaced with relative
placeholders (`py.txt`, `ray_results/`, `models--THUDM--chatglm3-6b`) and no
secrets were found in the scanned tree.
