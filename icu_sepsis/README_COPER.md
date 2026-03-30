## ICU-Sepsis bundle inside COPER

This folder vendors the useful ICU-Sepsis resources inside COPER so you can run
the MDP demo/metrics without relying on `code/icu-sepsis`:

- `icu_sepsis/` package (Gymnasium environment + bundled MDP assets)
- `icu_sepsis_helpers/` package (baseline metrics/helpers and rebuild tools)
- `assets/`, `examples/`, `demo_outputs/`
- `requirements.txt`, `README.md`, `LICENSE`
- `icu-sepsis-csv-tables.tar.gz`

Notebook copy:

- `code/COPER/notebooks/icu_sepsis_demo.ipynb`

### Local setup

From `code/COPER/`:

```bash
source .venv-coper/bin/activate
pip install -r requirements.txt
pip install -e icu_sepsis/icu_sepsis -e icu_sepsis/icu_sepsis_helpers
```

The notebook bootstraps local import paths automatically, so editable installs are
optional but recommended.

Notes:

- `icu-sepsis-csv-tables.tar.gz` is not needed to instantiate/run `Sepsis/ICU-Sepsis-v2`;
  the environment uses packaged assets from `icu_sepsis/.../envs/assets/`.
- Keep the archive if you want direct CSV access to transition/reward/initial-state/expert-policy tables.
