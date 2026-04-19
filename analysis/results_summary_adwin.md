# Resultados Preliminares — ADWIN Direcional

**Gerado em:** 2026-04-19  |  ARTE=21  NeuralARTE=80  SoftReset=66  HeteroBagging=46

**Configs selecionadas (melhor disponível por método):**

- ARTE: `mw10`
- NeuralARTE: `abc_proj_adwin_dir` → `abc_adwin_dir`
- SoftReset: `abc_proj_rl1_adwin` → `abc_rl1_adwin` → `abc_cnn_rl1_adwin`
- HeteroBagging: `abc_proj_adwin_dir` → `abc_adwin_dir` → `heterogeneous_adwin_dir`

## Accuracy — melhor config por método

| Dataset | ARTE | NeuralARTE | SoftReset | HeteroBagging | Melhor |
|---------|-----:|----------:|----------:|-------------:|--------|
| keystroke | — | — | — | — | — |
| ozone | 93.80% | 92.70% | 92.94% | **94.04%** | HB |
| outdoor | **85.62%** | 84.10% | 85.38% | 55.02% | ARTE |
| gassensor | 95.26% | 94.90% | **96.95%** | 92.89% | SR |
| electricity | 90.22% | **90.36%** | 89.48% | 88.51% | NR |
| shuttle | **99.70%** | 98.84% | 99.11% | 99.22% | ARTE |
| rialto | **84.10%** | 63.85% | 62.97% | 64.45% | ARTE |
| gmsc | **93.63%** | 90.99% | 90.91% | 93.52% | ARTE |
| covtype | **95.72%** | 93.16% | 89.72% | 90.56% | ARTE |
| airlines | 64.05% | 58.96% | 59.06% | **68.47%** | HB |
| sea_a | **89.73%** | — | 79.48% | 89.39% | ARTE |
| sea_g | **89.29%** | — | 79.17% | 89.01% | ARTE |
| mixed_a | **99.55%** | — | 97.86% | — | ARTE |
| mixed_g | **94.15%** | — | 84.78% | — | ARTE |
| led_a | **74.05%** | — | 71.95% | 73.97% | ARTE |
| led_g | 73.11% | — | 71.12% | **73.14%** | HB |
| agrawal_a | 84.41% | — | **87.89%** | — | SR |
| agrawal_g | — | — | **80.81%** | — | SR |
| rbf_f | **78.22%** | — | — | — | ARTE |
| rbf_m | **87.35%** | — | — | — | ARTE |
| **Wins** | **12** | **1** | **3** | **3** | |
| **Avg (disponível)** | 87.33% | 85.32% | 83.50% | 82.48% | |

## Datasets incompletos por método (adwin)

| Dataset | ARTE | NeuralARTE | SoftReset | HeteroBagging |
|---------|:----:|:----------:|:----------:|:-------------:|
| keystroke | **falta** | **falta** | **falta** | **falta** |
| sea_a | ✓ | **falta** | ✓ | ✓ |
| sea_g | ✓ | **falta** | ✓ | ✓ |
| mixed_a | ✓ | **falta** | ✓ | **falta** |
| mixed_g | ✓ | **falta** | ✓ | **falta** |
| led_a | ✓ | **falta** | ✓ | ✓ |
| led_g | ✓ | **falta** | ✓ | ✓ |
| agrawal_a | ✓ | **falta** | ✓ | **falta** |
| agrawal_g | **falta** | **falta** | ✓ | **falta** |
| rbf_f | ✓ | **falta** | **falta** | **falta** |
| rbf_m | ✓ | **falta** | **falta** | **falta** |
