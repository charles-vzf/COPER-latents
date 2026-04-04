# COPER — Continuous Patient state attention for irregular EHR time series

This directory is a **single working tree** that combines three upstream lines of work:

| Track | Role here | Upstream |
| --- | --- | --- |
| **COPER** | Irregular clinical time series → latent representations, mortality benchmarks, exportable bundles | [jmdvinodjmd/COPER](https://github.com/jmdvinodjmd/COPER) |
| **ICU-Sepsis** | Tabular sepsis MDP (Gymnasium), bundled dynamics for interpretation and RL | [icu-sepsis/icu-sepsis](https://github.com/icu-sepsis/icu-sepsis) · [ICU-Sepsis paper](https://arxiv.org/abs/2406.05646) |
| **Policy baselines** | SARSA, Q-learning, DQN, PPO, SAC on the ICU-Sepsis env (vendored experiment code) | [Dhawgupta/choudhary2024icu](https://github.com/Dhawgupta/choudhary2024icu) |

**MIMIC preprocessing** for mortality tasks is expected from an external clone of [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) (paths in `paths.json` / notebook configs).

---

## Repository layout

```
code/COPER/
├── README.md                 # this file (single source of truth for docs)
├── requirements.txt          # unified Python dependencies for COPER + notebooks + MDP
├── LICENSE
├── paths.json                # optional local path hints (MIMIC extracts, etc.)
├── src_coper/                # COPER core: attention, ODE cell, transformer baseline, losses
├── utils/                    # training entrypoint, export/load bundles, embeddings, viz
├── notebooks/                # analysis and demos (see below)
├── artifacts/                # exported model bundles (heavy files gitignored; metadata kept)
├── results/                  # training logs, tables, checkpoints (heavy patterns gitignored)
├── policies/                 # vendored RL algorithms (src/algos, experiments/, run/, analysis/)
│   └── (run from this dir: os.chdir in train_mdp_policies.ipynb)
└── icu_sepsis/               # vendored ICU-Sepsis environment + helpers
    ├── icu_sepsis/           # installable package: Gymnasium env + packaged dynamics.npz
    ├── icu_sepsis_helpers/   # value iteration, baselines, MDP rebuild utilities
    └── examples/             # quickstart, get_baselines, build_mimic_demo
```

**Notebooks (main entry points)**

| Notebook | Purpose |
| --- | --- |
| `compare_mortality_mimic3.ipynb` | Train/export COPER vs Transformer on MIMIC mortality |
| `display_embeddings.ipynb` | Latents from exported bundles |
| `latent_dim.ipynb` | Latent dimension sweep |
| `COPER_demo.ipynb` | Lightweight demo on local MIMIC extracts |
| `icu_sepsis_demo.ipynb` | MDP matrices, UMAP on state centers, rollouts |
| `coper_to_states.ipynb` | Map COPER latents toward MDP states |
| `train_mdp_policies.ipynb` | Train tabular RL policies; compare random, expert, **optimal** (value iteration), learned |
| `evaluations.ipynb` | Additional evaluations |

---

## ICU-Sepsis MDP: key formulas (paper-aligned)

States \(\mathcal{S}=\{0,\ldots,715\}\), actions \(\mathcal{A}^+=\{0,\ldots,24\}\), discount \(\gamma=1\) in the benchmark. Reward is terminal: +1 survival (state 714), 0 otherwise.

**Transition counts** from trajectories \(h\in\mathcal{D}\):

\[
C(s,a,s')=\sum_{h,t}\mathbf{1}\{S_t=s,A_t=a,S_{t+1}=s'\},\quad
C(s,a)=\sum_{s'} C(s,a,s').
\]

**Admissible actions** (threshold \(\tau\), e.g. 20): \(a\in\mathcal{A}(s)\) iff \(C(s,a)\ge\tau\). Inadmissible \((s,a)\) rows are replaced by the **average** of admissible transitions in \(s\) so the env stays a full \(|\mathcal{S}|\times|\mathcal{A}^+|\) table.

**Empirical transition and expert policy**

\[
p(s'|s,a)=\frac{C(s,a,s')}{C(s,a)}\quad\text{if }a\in\mathcal{A}(s),\qquad
\pi_{\mathrm{expert}}(a|s)=\frac{C(s,a)}{\sum_{\bar a}C(s,\bar a)}.
\]

**Initial distribution** \(d_0(s)\): empirical frequency of first state in each trajectory.

**Optimal policy**

Given \(p\) and \(R\), value iteration yields \(V^*\) and \(\pi^*(s)\in\arg\max_a Q^*(s,a)\). Implementation: `icu_sepsis_helpers.utils.mdp.value_iteration` (used in baselines and in `train_mdp_policies.ipynb`).

**Objective**

Expected return with \(\gamma=1\) equals survival probability for this reward design.

---

## COPER model (high level)

- **Input**: irregularly sampled multivariate clinical series (e.g. MIMIC benchmark tensors).
- **Core**: attention over continuous-time / irregular structure + optional ODE-inspired components (`src_coper/`).
- **Outputs**: task heads (e.g. in-hospital mortality) and **exportable latent bundles** for downstream analysis (`utils/export_coper_checkpoint.py`, `load_coper_bundle.py`, `coper_embed.py`).

Details are in the source modules and the original COPER reference repository.

---

## Setup

From `code/COPER/`:

```bash
python -m venv .venv-coper
source .venv-coper/bin/activate   # Windows: .venv-coper\Scripts\activate
pip install -r requirements.txt
```

Optional editable installs (notebooks can also `sys.path` the vendored trees):

```bash
pip install -e icu_sepsis/icu_sepsis
pip install -e icu_sepsis/icu_sepsis_helpers
```

**ICU-Sepsis assets**

- `Sepsis/ICU-Sepsis-v2` uses **packaged** dynamics under `icu_sepsis/icu_sepsis/.../envs/assets/`.
- The optional archive `icu-sepsis-csv-tables.tar.gz` is **not** committed (large); add locally if you need raw CSV tables. See upstream [icu-sepsis](https://github.com/icu-sepsis/icu-sepsis).

---

## Policy experiments (`policies/`)

This tree is a **copy** of the upstream experiment layout. Typical workflow:

```bash
cd policies
python src/mainjson.py experiments/debug.json 0
```

Sweep / analysis / plots (when you use the full upstream pipeline):

```bash
python run/local.py -p src/mainjson.py -j experiments/debug.json
python analysis/process_data.py experiments/debug.json
python analysis/learning_curve.py y returns auc experiments/debug.json
```

For day-to-day work inside this monorepo, **`notebooks/train_mdp_policies.ipynb`** is the main entry: it `chdir`s to `policies/` and calls `src/algos/` (SARSA, Q-learning, DQN, PPO, SAC) on `Sepsis/ICU-Sepsis-v2`, plus **optimal** via value iteration.

---

## References

```bibtex
@inproceedings{choudhary2024icusepsis,
  title={{ICU-Sepsis}: A Benchmark {MDP} Built from Real Medical Data},
  author={Choudhary, Kartik and Gupta, Dhawal and Thomas, Philip S.},
  booktitle={Reinforcement Learning Conference},
  year={2024},
  url={https://arxiv.org/abs/2406.05646}
}
```

- Clinical pipeline lineage: [Komorowski et al., Nature Medicine 2018](https://www.nature.com/articles/s41591-018-0213-5) (AI Clinician / MIMIC sepsis cohort).

---

## Git / hygiene

- **One** `.gitignore` at the root of `code/COPER/`; ignore heavy `artifacts/`, `results/`, checkpoints, optional CSV archives, and local venvs.
- Subfolders no longer ship their own `README.md` or `.gitignore`; documentation lives here.

## License

See `LICENSE` in this directory.
