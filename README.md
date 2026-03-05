# KD-AnomalyNet: What Knowledge Transfers in Tabular Anomaly Detection?

> **A Teacher–Student Distillation Analysis**  
> *Published in Machine Learning and Knowledge Extraction (MDPI), 2026*

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fmake8030060-blue)](https://doi.org/10.3390/make8030060)
[![Journal](https://img.shields.io/badge/Journal-MAKE%202026-green)](https://www.mdpi.com/journal/make)
[![License: CC BY](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

Paper:

**Krčmar, T., Šabanović, D., Švarcmajer, M., & Lukić, I. (2026).**  
*What Knowledge Transfers in Tabular Anomaly Detection? A Teacher–Student Distillation Analysis.*  
Machine Learning and Knowledge Extraction, 8(3), 60.  
**[https://doi.org/10.3390/make8030060](https://doi.org/10.3390/make8030060)**

---

## Overview

This repository contains the code and experiments for **KD-AnomalyNet**, a teacher–student knowledge distillation framework for efficient tabular anomaly detection.

Rather than treating distillation as purely performance-driven compression, we investigate **what anomaly knowledge transfers** and **what gets lost** during the process. A key contribution is a **diffusion-inspired perturbation analysis** used as a diagnostic probe for representation stability — without introducing additional trainable components.

### Key Results

| Metric | Value |
|--------|-------|
| Teacher AUC-ROC retention | up to **98.5%** |
| Inference speedup | **26–181×** |
| Student parameters | < 8,000 |
| Student ranking (vs. 16 baselines) | **4th of 16** |

### What Transfers Well vs. What Degrades

| Knowledge Type | Transfer Rate |
|---|---|
| Isolation-based detection (IF) | **88.3%** retention |
| Global outlier detection | **78.0%** transfer |
| Neighborhood-based detection (LOF) | 76.0% retention |
| Local outlier detection | only **20.0%** transfer |

---

## Method

The framework consists of three stages:

1. **Teacher Training** — A high-capacity ensemble of four detectors (Autoencoder, VAE, Isolation Forest, LOF) is trained on normal data.
2. **Student Distillation** — A lightweight dual-head MLP student is trained using a boundary-aware distillation loss with a temperature curriculum (exponential decay from T₀=5.0 to T_f=1.0).
3. **Deployment** — Only the student is used at inference time.

The **diffusion-inspired perturbation analysis** applies controlled Gaussian noise at multiple scales (σ ∈ {0.01, 0.05, 0.1, 0.2, 0.5}) to probe the stability of teacher and student anomaly representations without learning a denoising model.

---

## Datasets

Experiments are conducted on **10 benchmarks** from the [ODDS repository](http://odds.cs.stonybrook.edu/):

`cardio` · `satellite` · `thyroid` · `mammography` · `annthyroid` · `pima` · `pendigits` · `optdigits` · `vowels` · `musk`

Datasets span dimensionalities d ∈ [6, 166] and anomaly rates from 2.3% to 35.1%.

---

## Configuration

Key hyperparameters (see `default.yaml`):

| Parameter | Value |
|-----------|-------|
| Loss balance α | {0.1, 0.3, 0.5, 0.7, 0.9} |
| Boundary margin δ_b | 0.1 |
| Boundary weight w_b | 2.0 |
| Initial temperature T₀ | 5.0 |
| Final temperature T_f | 1.0 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Schedule | Cosine annealing + 10-epoch warmup |

---

## Citation

If you use this code or findings in your research, please cite:

**APA**
```
Krčmar, T., Šabanović, D., Švarcmajer, M., & Lukić, I. (2026). What Knowledge Transfers in Tabular Anomaly Detection? A Teacher–Student Distillation Analysis. Machine Learning and Knowledge Extraction, 8(3), 60. https://doi.org/10.3390/make8030060
```

**BibTeX**
```bibtex
@article{krcmar2026knowledge,
  title     = {What Knowledge Transfers in Tabular Anomaly Detection? A Teacher--Student Distillation Analysis},
  author    = {Kr{\v{c}}mar, Tea and {\v{S}}abanovi{\'c}, Dina and {\v{S}}varcmajer, Miljenko and Luki{\'c}, Ivica},
  journal   = {Machine Learning and Knowledge Extraction},
  volume    = {8},
  number    = {3},
  pages     = {60},
  year      = {2026},
  publisher = {MDPI},
  doi       = {10.3390/make8030060},
  url       = {https://doi.org/10.3390/make8030060}
}
```

**MDPI / ACS**
```
Krčmar, T.; Šabanović, D.; Švarcmajer, M.; Lukić, I. What Knowledge Transfers in Tabular Anomaly Detection? A Teacher–Student Distillation Analysis. Mach. Learn. Knowl. Extr. 2026, 8, 60. https://doi.org/10.3390/make8030060
```

**Chicago / Turabian**
```
Krčmar, Tea, Dina Šabanović, Miljenko Švarcmajer, and Ivica Lukić. 2026. "What Knowledge Transfers in Tabular Anomaly Detection? A Teacher–Student Distillation Analysis" Machine Learning and Knowledge Extraction 8, no. 3: 60. https://doi.org/10.3390/make8030060
```

---

## Affiliation

Faculty of Electrical Engineering, Computer Science and Information Technology Osijek  
Josip Juraj Strossmayer University of Osijek, Croatia  
📧 tea.krcmar@ferit.hr

---

## License

This work is published open access under the [Creative Commons Attribution (CC BY) 4.0](https://creativecommons.org/licenses/by/4.0/) license.
