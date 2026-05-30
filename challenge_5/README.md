# Challenge 5 — Group 6: Housing & Urban Development

Short summary
-------------
This folder contains the code and artifacts for Challenge 5 (Group 6) on housing data clustering. You'll find the main script, configuration files, and outputs produced by runs.

Main structure
--------------
- **Main script**: [challenge_5/challenge5_grupo6.py](challenge_5/challenge5_grupo6.py#L1) — the pipeline script that runs download, cleaning, EDA, dimensionality reduction, clustering grid searches and saves figures/tables.
- **Configuration**: [challenge_5/config.json](challenge_5/config.json#L1) — default configuration; pass an alternative config with `--config`.
- **Project metadata / dependencies**: [challenge_5/pyproject.toml](challenge_5/pyproject.toml#L1).
- **Checklist**: [challenge_5/CHECKLIST.md](challenge_5/CHECKLIST.md#L1)
- **Runs / outputs**: [challenge_5/runs/](challenge_5/runs) — per-run folders containing saved results (figures, tables, and `config_used.json`).

How to run
----------
From the `challenge_5` folder (or specifying the path), run:

```bash
python challenge5_grupo6.py --config config.json
```

Environment & setup
-------------------
It is recommended to run the project inside a virtual environment. Example steps (bash/zsh):

```bash
# create and activate a venv
python -m venv .venv
source .venv/bin/activate

# upgrade pip and install dependencies
pip install --upgrade pip
# If a requirements.txt exists:
pip install -r requirements.txt

# Alternatively, if you use Poetry and the project provides a pyproject.toml:
poetry install

# To generate a requirements.txt from an active env:
pip freeze > requirements.txt
```

Useful notes
------------
- The script writes outputs into `runs/<timestamp>/` (figures in `figures/`, tables in `results/`).
- If `pyproject.toml` is present you can inspect it for dependency metadata or use it with your preferred tool (e.g. `poetry` or static inspection).
- Check here [challenge_5/CHECKLIST.md](challenge_5/CHECKLIST.md#L1)

Authors
----------------
Nahin Peñaranda
Samuel Casas
Giovanni Vargas
