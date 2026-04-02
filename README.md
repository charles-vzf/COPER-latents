# Studying COntinuous Patient state attention model for addressing irregularity in Electronic health Records (COPER)

## References

- **COPER**: [jmdvinodjmd/COPER](https://github.com/jmdvinodjmd/COPER)
- **MIMIC-III benchmark preprocessing**: [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)
- **ICU-Sepsis MDP**: [icu-sepsis/icu-sepsis](https://github.com/icu-sepsis/icu-sepsis)

## Overview

This repository studies **COPER** as a patient representation model for irregular clinical time series.
The main focus is:

- training and comparing COPER variants on **MIMIC-III in-hospital mortality**
- exporting trained models as reusable bundles
- extracting and visualizing latent embeddings
- interpreting COPER embeddings against the vendored **ICU-Sepsis** tabular MDP

The repo is centered on the local `code/COPER` project, with MIMIC preprocessing expected to come from an external `mimic3-benchmarks` setup.

## Repository Structure

### Core model code

- `src_coper/`: COPER, Transformer baseline, ODE cell, attention blocks, metrics, and dataset utilities
- `utils/run_exp.py`: main training entry point
- `utils/export_coper_checkpoint.py`: exports trained checkpoints to portable bundles
- `utils/load_coper_bundle.py`: reloads exported bundles for inference / embedding extraction
- `utils/coper_embed.py`: extracts representations before the classifier
- `utils/embedding_data_utils.py`: helpers for loading mortality splits and batching data
- `utils/viz_utils.py`: utilities for latent-space visualization

### Notebooks

- `notebooks/compare_mortality_mimic3.ipynb`: benchmark notebook for training and exporting COPER / Transformer models on MIMIC mortality
- `notebooks/display_embeddings.ipynb`: computes and visualizes latent embeddings from exported bundles
- `notebooks/latent_dim.ipynb`: latent-dimension sweep for COPER
- `notebooks/COPER_demo.ipynb`: lightweight COPER demo on local raw MIMIC extracts
- `notebooks/icu_sepsis_demo.ipynb`: explores the ICU-Sepsis MDP and its bundled assets
- `notebooks/coper_to_states.ipynb`: learns a small head from COPER embeddings to MDP-state distributions for interpretation

### ICU-Sepsis integration

- `icu_sepsis/`: vendored ICU-Sepsis code and assets used for MDP-based interpretation
- `icu_sepsis/icu_sepsis/`: environment package
- `icu_sepsis/icu_sepsis_helpers/`: helper code for analysis, baselines, and MDP construction

### Outputs

- `artifacts/`: exported model bundles and saved latent-analysis artifacts
- `results/`: run outputs such as tables, temporary checkpoints, logs, and demo model files