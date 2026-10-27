# Contributing

This is a dormant research archive, not a maintained package. Corrections and
reproducibility fixes are welcome; large rewrites are probably a better fit for
a fork.

## Before you start

- Open an issue first for anything beyond a typo or a broken link. Name the
  notebook and cell, and say what you expected to happen.
- Keep the archive faithful to what was run. In particular, do not rename
  notebooks or rewrite their imports to `cznn`; the local `nn.py` is there so
  the original `from nn import *` lines keep working.
- Do not commit corpora, checkpoints, `*.map` files, `ray_results/`, notebook
  outputs, or generated logs. They are ignored on purpose.
- If a change adds a dependency, say so in the pull request and update the
  dependency list in `README.md`. There is no requirements file or lockfile.

## Workflow

1. Fork the repository and create a focused branch.
2. Make the smallest change that addresses the issue.
3. If you changed a notebook, describe the cell and how you verified it. A
   static correction does not need a full training run.
4. Open a pull request with the reasoning and the files touched.

## Scope

Reasonable contributions include: fixing a broken relative path, correcting a
description that no longer matches the code, documenting a missing dependency,
or adding a small reproducible example. New model research belongs in a fork
unless it directly continues the retention experiments already here.

## License

By contributing, you agree that your contribution is licensed under
[AGPL-3.0-only](LICENSE), the same license as the rest of the repository.
