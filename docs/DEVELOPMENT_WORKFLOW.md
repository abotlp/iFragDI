# Development Workflow

## Roles

- Patricia runs tests, Slurm jobs, environment checks, and validates outputs.
- Codex makes scoped code/documentation changes only after explicit prompts.
- ChatGPT supervises scientific logic, code diffs, and next-step decisions.

## Branch Policy

- `main` = reviewed working snapshot / stable-enough baseline.
- `codex/*` branches = proposed changes.
- No direct unreviewed changes to `main` after setup.

## Execution Policy

- Codex must not submit Slurm jobs.
- Codex must not run full benchmark jobs.
- Codex may run only lightweight syntax checks/static tests when explicitly requested.
- Patricia manually runs computational tests.

## Benchmark Data Policy

- Raw BM5 files, PDBs, generated outputs, matrices, images, logs, and databases are not committed.
