[🇨🇳 中文版](README-cn.md)
# HVG–Autoencoder–SINDy: Chaotic Dynamics of Cultural Attention

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: 2014–2024](https://img.shields.io/badge/data-2014--2024-teal.svg)]()
[![Status: Preprint](https://img.shields.io/badge/status-preprint-orange.svg)]()

Official implementation of **"Chaotic Dynamics of Tai Chi Public Attention Revealed by an Integrated Framework of Horizontal Visibility Graphs, Autoencoders, and Sparse Identification"**.

Using daily Baidu Index data (2014–2024) from four Chinese provinces (Beijing, Shanghai, Guangdong, Henan), this pipeline characterizes low-dimensional chaotic attractors, discovers sparse governing equations, and benchmarks against ARIMA, VAR, and LSTM baselines.

---

## Overview

The framework operates as a sequential analytical pipeline:

```
Time series → HVG module → Autoencoder → SINDy (STLS) → Lyapunov / Phase space
              (12-dim features) (3-dim latent)  (sparse equations)  (validation)
```

**Key components:**
- **HVG module** — constructs Horizontal Visibility Graphs and extracts 12-dimensional topological feature vectors via a sliding window; estimates chaos criterion λ vs. ln(3/2) ≈ 0.405
- **Autoencoder** — fully connected encoder [12→64→32→16→3] with symmetric decoder; VAE variant included for comparison
- **SINDy (STLS)** — discovers sparse discrete-time governing equations $z(t+1) = f(z(t))$ from a 48-term library (constant, linear, quadratic, trigonometric)
- **Baselines** — ARIMA, VAR, LSTM applied in the same latent space for fair comparison
- **Chaos diagnostics** — Rosenstein and Wolf Lyapunov exponents, phase portraits, Poincaré sections, correlation dimension

The form of the governing equation is as follows:

$$
\begin{bmatrix} z_1(t+1) \\ z_2(t+1) \\ z_3(t+1) \end{bmatrix} ^\top= C + A \underbrace{\mathbf{z}}_{\text{linear}} + \underbrace{\begin{bmatrix}
    \mathbf{z}^\top Q_1 \mathbf{z} \\
    \mathbf{z}^\top Q_2 \mathbf{z} \\
    \mathbf{z}^\top Q_3 \mathbf{z}
\end{bmatrix}}_{\text{quadratic}} + D \underbrace{\boldsymbol{\phi}}_{\text{trigonometric}}
$$

where

$$
\mathbf{z} = \begin{bmatrix} z_1 & z_2 & z_3 \end{bmatrix}^\top , \boldsymbol{\phi} = \begin{bmatrix} \sin z_1 & \cos z_1 & \sin z_2 & \cos z_2 & \sin z_3 & \cos z_3 \end{bmatrix}^\top
$$


---

## Key Results

| Metric | Value |
|--------|-------|
| HVG λ range | 0.257 – 0.437 (reference: ln(3/2) ≈ 0.405) |
| AE reconstruction correlation | 0.896 – 0.918 (VAE: 0.821–0.823) |
| SINDy NRMSE (in-sample) | 11.7 – 12.8% |
| Active SINDy terms | 30–38 / 48 candidates |
| Rosenstein λ_max | 0.025 – 0.036 bits/day |
| Predictability horizon | 30 – 70 days |

### Regional chaos classification

| Province | Channel | HVG λ | λ_max (Rosenstein) | λ_max (Wolf) | Regime |
|----------|---------|-------|--------------------|--------------|--------|
| Beijing  | PC      | 0.291 | 0.027 | 0.171 | Strong chaos |
| Beijing  | Mobile  | 0.316 | 0.033 | 0.179 | Strong chaos |
| Guangdong | PC     | 0.376 | 0.026 | 0.020 | Strong chaos |
| Guangdong | Mobile | 0.376 | 0.034 | 0.174 | Strong chaos |
| Shanghai | Mobile  | 0.257 | 0.032 | 0.178 | Strong chaos |
| Henan    | PC      | 0.415 | 0.026 | 0.016 | Quasi-periodic |
| Henan    | Mobile  | 0.434 | 0.036 | 0.217 | Quasi-periodic |
| Shanghai | PC      | 0.437 | 0.025 | 0.195 | Quasi-periodic |

### Comparison with baselines (50-day test, Mobile channel, NRMSE %)

| Method | Beijing | Guangdong | Henan | Shanghai |
|--------|---------|-----------|-------|----------|
| ARIMA  | 21.1    | 24.2      | 21.0  | 18.2     |
| VAR    | 21.2    | 24.7      | 21.5  | 18.7     |
| LSTM   | 31.1    | 28.9      | 24.2  | 25.9     |
| **SINDy (ours)** | **20.8** | **22.4** | **19.9** | **17.6** |

---

## Project Structure

```
HVG-SINDy-Autoencoder/
├── data/                  # Baidu Index Excel files
│   ├── Beijing.xlsx
│   ├── Guangdong.xlsx
│   ├── Henan.xlsx
│   └── Shanghai.xlsx
├── results/               # Output figures, tables, JSON reports
├── hvg_sindy.py           # Core: HVG construction, AE/VAE, SINDy, Lyapunov
├── comparison.py          # Baseline methods: ARIMA, VAR, LSTM
├── data_sequence.py       # temporal pattern visualization
├── main.py                # Entry point
└── requirements.txt
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/wind-whispered/HVG-SINDy-Autoencoder.git
cd HVG-SINDy-Autoencoder

# Install dependencies
pip install -r requirements.txt
pip install openpyxl
```

**Requirements:** Python >= 3.8, PyTorch >= 2.0, CUDA optional.

Core packages:

```
torch>=2.0
numpy
pandas
scipy
networkx          # HVG construction
statsmodels       # ARIMA / VAR baselines
scikit-learn      # preprocessing and metrics
matplotlib
seaborn
openpyxl          # Excel file reading
```

---

## Data Preparation

Place province Excel files in `./data/`. Each file should contain three columns:

| Column | Format | Description |
|--------|--------|-------------|
| `date` | YYYYMMDD | Daily date |
| `PC`   | integer | PC-based search index |
| `Mobile` | integer | Mobile-based search index |

Files must be named `Beijing.xlsx`, `Guangdong.xlsx`, `Henan.xlsx`, `Shanghai.xlsx`. The dataset covers 4,018 daily observations per province per channel (January 1, 2014 – December 31, 2024), yielding 32,144 total observations.

---

## Usage

Run the full pipeline (HVG analysis → autoencoder training → SINDy identification → comparative evaluation → all figures and tables):

```bash
python main.py
```

All outputs are saved to `./results/`:
- `figure_*.png / .pdf` — all paper figures
- `Table*.csv` — descriptive statistics, chaos metrics, AE performance, SINDy coefficients
- `analysis_results.json` — full numerical results

Individual scripts can also be run independently:

```bash
# Temporal pattern visualization
python data_sequence.py

# Comparative analysis only (requires saved SINDy models from main.py)
python comparison.py
```

---

## SINDy Coefficient Structure

Representative governing equations for Beijing (3-dimensional latent space):

| Province | Linear (mean) | Quadratic (mean) | Trigonometric (mean) | Active terms | NRMSE (%) |
|----------|--------------|-----------------|---------------------|-------------|-----------|
| Beijing  | 0.337 | 0.150 | 0.500 | 37/48 | 12.7 |
| Guangdong | 0.398 | 0.111 | 0.354 | 38/48 | 11.9 |
| Henan    | 0.222 | 0.115 | 0.287 | 38/48 | 12.8 |
| Shanghai | 0.240 | 0.092 | 0.349 | 30/48 | 11.7 |

---

## Citation

```bibtex
@article{kang2026taichi,
  title   = {Chaotic Dynamics of Tai Chi Public Attention Revealed by an Integrated
             Framework of Horizontal Visibility Graphs, Autoencoders, and Sparse
             Identification},
  author  = {Kang, Yafeng and Li, Pengchao and Tang, Lu and Zhang, Chao},
  year    = {2026},
  note    = {Preprint}
}
```

---

## Reference

- Brunton et al. (2016) — original SINDy framework ([PySINDy](https://github.com/dynamicslab/pysindy))

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. Data sourced from Baidu Index; users are responsible for compliance with Baidu's terms of service.

---

