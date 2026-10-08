# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

- Python >=3.14, managed with uv. Use `uv sync`, `uv add <pkg>` (`--dev` for tooling) and `uv run <cmd>`; don't use pip directly.

## Commands

- Test: `uv run pytest`; single test: `uv run pytest path/to/test_file.py::test_name`
- Lint: `uv run ruff check --fix`
- Format: `uv run ruff format`
