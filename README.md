# Continuous patient state attention model for addressing irregularity in electronic health records

This repository contains the implementation of **COPER** and the training scripts used with
the **preprocessed** MIMIC-III mortality dataset and **PhysioNet Challenge 2012**.

**Upstream reference code** (original release accompanying the papers): [github.com/jmdvinodjmd/COPER](https://github.com/jmdvinodjmd/COPER).

## Setup goal

The purpose of this codebase is to study the quality of the learned **patient embeddings**
and to experiment with architectural variants (e.g. the 1-NODE vs 2-NODE COPER variants).

## Data preprocessing repositories

* MIMIC-III / mortality preprocessing: [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)
* PhysioNet Challenge 2012: [PhysioNet Challenge 2012 (1.0.0)](https://physionet.org/content/challenge-2012/1.0.0/)

Code for the papers:
> [Chauhan, V. K., Thakur, A., O’Donoghue, O., Rohanian, O., Molaei, S., & Clifton, D. A. (2024). Continuous patient state attention model for addressing irregularity in electronic health records. BMC Medical Informatics and Decision Making, 24(1), 117.](https://link.springer.com/article/10.1186/s12911-024-02514-2)

> [Chauhan, V. K., Thakur, A., O'Donoghue, O., & Clifton, D. A. (2022). COPER: Continuous patient state perceiver. In 2022 IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI) (pp. 1-4). IEEE.](https://ieeexplore.ieee.org/document/9926807)

## Python environment (venv)

We provide:

* `requirements.txt` for non-PyTorch dependencies
* `scripts/setup_venv.sh` to recreate `.venv-coper/` and install PyTorch + requirements

To recreate the venv:

```bash
cd code/COPER
./scripts/setup_venv.sh
```

Activate:

```bash
source .venv-coper/bin/activate
```

Note: `scripts/setup_venv.sh` installs a CUDA build of `torch` when `nvidia-smi` is available,
otherwise it installs the CPU build.

To keep an existing venv, run `./scripts/setup_venv.sh --keep`.

## Experiment

You need mimic-iii [mimic-iii](https://github.com/YerevaNN/mimic3-benchmarks) and [Physionet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) datasets.

Training entry point (from `code/COPER/`):

```bash
python utils/run_exp.py --help
```

To run different experiments, you can use the following shell scripts:
* ```scripts/experiments/run_irregular_mimic.sh```: to study irregularity on mimic dataset.
* ```scripts/experiments/run_irregular_physionet.sh```: to study irregularity on physionet dataset.
* ```scripts/experiments/run_exp_normal.sh```: to run Perceiver and baselines (without irregularity).
* ```scripts/experiments/run_perceiver_latents.sh```: to study the effect of number of latents on the Perceiver and compare with Transformer.

## Embedding quality study (MIMIC-III mortality + 1-NODE vs 2-NODE)

The `notebooks/` folder contains an end-to-end benchmark designed to:
1. Train COPER under two architectural variants (flag `--second-node`).
2. Export trained checkpoints to portable bundles (`.pt` + `.json`).
3. Compute latent embeddings and visualize/compare them.

Key notebooks:

* `notebooks/compare_mortality_mimic3.ipynb`: runs the benchmark and saves tables under `results/tables/`.
* `notebooks/latent_dim.ipynb`: **1-NODE** COPER only — sweeps `--latent-dim` on MIMIC mortality, records test metrics and parameter counts, plots performance vs latent size / model size.
* `notebooks/display_copers_embeddings.ipynb`: loads exported bundles and visualizes embeddings (PCA / UMAP / t-SNE).
* `notebooks/demo.ipynb`: fast end-to-end demo using COPER on raw local MIMIC-III CSV extracts (no benchmark pickle dependency).

Export helper:

* `utils/export_coper_checkpoint.py`
* `utils/load_coper_bundle.py`

Outputs are stored under:
* `results/` (checkpoints/logs/predictions via `--results-dir`)
* `artifacts/` (portable embedding bundles)

`.gitignore` excludes heavy run artifacts (venv, `__pycache__/`, `results/logs/`, `results/checkpoints/`, `Predictions_*.npz`, etc.) while leaving small summaries such as `results/tables/*.csv` trackable if you commit them.

## Citations:
```
@article{chauhan2024continuous,
  title={Continuous patient state attention model for addressing irregularity in electronic health records},
  author={Chauhan, Vinod Kumar and Thakur, Anshul and O’Donoghue, Odhran and Rohanian, Omid and Molaei, Soheila and Clifton, David A},
  journal={BMC Medical Informatics and Decision Making},
  volume={24},
  number={1},
  pages={117},
  year={2024},
  publisher={Springer}
}

@inproceedings{Chauhan2022a,
  title={COPER: Continuous Patient State Perceiver},
  author={Chauhan, Vinod Kumar and Thakur, Anshul and O'Donoghue, Odhran and Clifton, David A},
  booktitle={IEEE International Conference on Biomedical and Health Informatics},
  year={2022},
  url={https://arxiv.org/abs/2208.03196}
}

@misc{coper_repo,
  title={COPER repository},
  howpublished={\url{https://github.com/jmdvinodjmd/COPER}},
  year={2024},
}

@misc{mimic3_benchmarks,
  title={mimic3-benchmarks (MIMIC-III benchmark preprocessing)},
  howpublished={\url{https://github.com/YerevaNN/mimic3-benchmarks}},
  year={2024},
}

@misc{physionet2012,
  title={PhysioNet Challenge 2012 (1.0.0)},
  howpublished={\url{https://physionet.org/content/challenge-2012/1.0.0/}},
  year={2012},
}
```

Neural ODE implementations based on [[Yulia Rubanova]](https://github.com/YuliaRubanova/latent_ode).
