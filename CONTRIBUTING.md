# Contributing to faster-whisper

Contributions are welcome! Here are some pointers to help you install the library for development and validate your changes before submitting a pull request.

## Install the library for development

We recommend using [uv](https://docs.astral.sh/uv/) with the `dev` extra to install the module in editable mode with pinned, reproducible dependencies (`uv.lock`):

```bash
git clone https://github.com/SYSTRAN/faster-whisper.git
cd faster-whisper/
uv sync --extra dev
```

## Validate the changes before creating a pull request

1. Make sure the existing tests are still passing (and consider adding new tests as well!):

```bash
uv run pytest tests/
```

2. Lint and format the code with [ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff check .
uv run ruff format .
```

These steps are also run automatically in the CI when you open the pull request.
