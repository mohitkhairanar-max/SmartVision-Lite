# Repository Guidelines

## Project Structure & Module Organization

- Main application: `app.py` (Streamlit) and `webcam_demo.py` (OpenCV)
- Core detection logic: `detection/` directory (detector, processors, metrics, utilities)
- Configuration: `config/` directory
- Model weights: `models/` directory (Git-ignored)
- Sample data: `sample_data/` directory (Git-ignored)
- Outputs: `outputs/` directory (Git-ignored)
- Tests: `tests/` directory
- Training: `training/` directory (Colab notebook)
- Utilities: `scripts/` directory (e.g., smoke test)

## Build, Test, and Development Commands

- Install dependencies: `pip install -r requirements.txt` (or use `pyproject.toml`)
- Set up virtual environment: `python -m venv .venv` then activate
- Run Streamlit app: `streamlit run app.py`
- Run desktop webcam demo: `python webcam_demo.py`
- Run tests: `pytest` (see `.github/workflows` for CI)
- Smoke test (requires model): `python scripts/smoke_test.py --device cpu`
- Install in development mode: `pip install -e .` (from pyproject.toml)

## Coding Style & Naming Conventions

- Follow PEP 8 for Python code (4-space indentation)
- Naming: snake_case for functions/variables, PascalCase for classes
- Constants: UPPER_SNAKE_CASE
- Docstrings: Use triple double quotes, following the style in the codebase
- Type hints: Use as shown in the source code (e.g., `-> str`, `: int`)

## Testing Guidelines

- Testing framework: pytest
- Test file location: `tests/` directory
- Test naming: `test_*.py` or `*_test.py` (follow existing tests)
- Run tests: `pytest` (unit tests) or `python scripts/smoke_test.py` (real-model smoke test)
- Coverage: Aim to maintain existing coverage; new features should include tests
- Markers: Use `@pytest.mark.integration` for tests that require model download or hardware

## Commit & Pull Request Guidelines

- Commit messages: Use clear, descriptive messages (imperative mood, e.g., "Add feature X")
- Pull requests: Must include a summary of changes, reference related issues, and include screenshots for UI changes
- Code review: Require at least one approval from maintainers
- CI: Ensure all GitHub Actions checks pass before merging
