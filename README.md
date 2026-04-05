# river-exp

Implementação e experimentação do algoritmo **ARTE** (*Adaptive Random Tree Ensemble*, Paim & Enembreck) em Python usando a biblioteca [River](https://riverml.xyz), com variantes neurais (NeuralARTE) e baseadas em GNN.

---

## Estrutura do projeto

```
river-exp/
├── src/
│   ├── shared/
│   │   ├── metrics.py          # KappaM e outras métricas auxiliares
│   │   └── utils.py            # Carregamento de datasets (ARFF) e logging CSV
│   ├── arte/
│   │   ├── drift_detector.py   # ADWINChangeDetector (wrapper com checagem de direção)
│   │   ├── splitter.py         # ARTEGaussianSplitter (ponto de corte aleatório, fiel ao Java)
│   │   ├── nodes.py            # RandomSubspaceNodeMixin + nós folha (MC, NB, NBA)
│   │   ├── tree.py             # ARTEHoeffdingTree
│   │   └── ensemble.py         # ARTE (ensemble completo)
│   ├── neural_arte/
│   │   └── neural_arte.py      # NeuralARTE (MLPs via deep-river, Online Bagging)
│   └── gnn/
│       ├── ensemble_gnn.py     # MetaGNNAggregator (GAT/MLP para agregação do ensemble)
│       └── gnn_arte.py         # GNN-ARTE (NeuralARTE + Meta-GNN)
│
├── experiments/
│   ├── arte/
│   │   ├── run_experiments.py      # Runner principal do ARTE (CLI)
│   │   ├── run_all.sh              # Dispara todos os datasets em paralelo (screen)
│   │   ├── run_mw_comparison.sh    # Compara ADWIN mw=5 vs mw=10
│   │   └── run_adwin_direction.sh  # Valida ADWINChangeDetector com checagem de direção
│   ├── neural_arte/
│   │   ├── run_all.sh              # Todos os datasets NeuralARTE
│   │   └── run_ablation.sh         # Ablação de variantes (camadas, etc.)
│   └── gnn/
│       └── run_experiments.sh      # GNN-ARTE (electricity, sea_g, outdoor) em 2 GPUs
│
├── analysis/
│   ├── compare_mw.py           # Tabela comparativa mw=5 vs mw=10
│   ├── results_arte.py         # Sumário dos resultados ARTE
│   ├── results_neural.py       # Sumário NeuralARTE
│   ├── results_gnn.py          # Sumário GNN-ARTE
│   └── compare_ablation.py     # Ablação NeuralARTE
│
├── java_src/                   # Código Java original do MOA (referência) + gerador de datasets
│   └── generate_moa_datasets.sh
│
├── notebooks/                  # Jupyter notebooks exploratórios
└── results/
    ├── arte/                   # CSVs dos experimentos ARTE
    ├── adwin_direction/        # CSVs dos testes de direção ADWIN
    ├── gnn/                    # CSVs dos experimentos GNN-ARTE
    └── logs/                   # Logs dos processos screen
```

---

## Variáveis de ambiente

Todos os scripts de experimento respeitam estas variáveis. Não é necessário editar nenhum arquivo.

| Variável | Padrão (hardcoded em utils.py) | Descrição |
|---|---|---|
| `DATASETS_PATH` | `/home/charan/moa/aldopaim/AdaptiveRandomTreeEnsemble/datasets` | Pasta com os ARFFs dos datasets |
| `PYTHON` | `python3` | Interpretador Python a usar |

Exemplos de uso:
```bash
# Exportar antes de rodar qualquer script
export DATASETS_PATH=/home/marcelo.charan1/Documents/moa/AdaptiveRandomTreeEnsemble/datasets
export PYTHON=/home/marcelo.charan1/.conda/envs/deep-river-demo/bin/python
bash experiments/arte/run_all.sh

# Ou passar diretamente via argumento ao runner Python
python experiments/arte/run_experiments.py \
    --dataset electricity \
    --datasets_path /home/marcelo.charan1/Documents/moa/AdaptiveRandomTreeEnsemble/datasets
```

---

## 1. Gerar datasets sintéticos (MOA)

Os datasets sintéticos são gerados via MOA CLI. O script está em `java_src/generate_moa_datasets.sh`.

```bash
# Modo completo: 1M instâncias, 3 drifts a cada 250k (usado nos experimentos)
bash java_src/generate_moa_datasets.sh --mode full --output-dir /caminho/datasets

# Modo reduzido: 50k instâncias (fator 1/20 — para testes rápidos)
bash java_src/generate_moa_datasets.sh --mode reduced --output-dir /caminho/datasets/mini

# Especificando a pasta da lib do MOA (se diferente do padrão)
bash java_src/generate_moa_datasets.sh --moa-dir /caminho/moa/lib --output-dir /caminho/datasets
```

Datasets gerados: `agrawal_a/g`, `led_a/g`, `sea_a/g`, `mixed_a/g`, `rbf_f`, `rbf_m`.

> **Nota sobre RBF no modo reduced:** a velocidade de deriva do RBF *não* é escalonada proporcionalmente, para evitar que os conceitos sejam inlearnable num stream curto.

---

## 2. Executar experimentos ARTE

### Rodada completa (todos os datasets em paralelo via `screen`)

```bash
# A partir da raiz do projeto
bash experiments/arte/run_all.sh
```

Cada dataset roda numa sessão `screen` separada (`exp_<dataset>`). Acompanhe com:
```bash
screen -ls | grep exp_
tail -f results/logs/...  # logs individuais dos scripts que os geram
```

### Dataset individual

```bash
python experiments/arte/run_experiments.py \
    --dataset rbf_m \
    --seed 123456789 \
    --n_models 100 \
    --window_size 500 \
    --adwin_min_window 5 \
    --datasets_path /caminho/datasets
```

Parâmetros:

| Argumento | Padrão | Descrição |
|---|---|---|
| `--dataset` | obrigatório | Nome do dataset (ver lista em `src/shared/utils.py`) |
| `--seed` | `123456789` | Semente aleatória |
| `--n_models` | `100` | Número de árvores no ensemble |
| `--window_size` | `500` | Janela deslizante para acurácia por membro |
| `--adwin_min_window` | `5` | `min_window_length` do ADWIN (`5` = padrão River, `10` = equivalente MOA) |
| `--datasets_path` | valor de `DATASETS_PATH` | Sobrescreve a variável de ambiente |

Saída: `results/arte/ARTE_CPU_{dataset}_mw{mw}_s{seed}_{timestamp}.csv`

### Comparação mw=5 vs mw=10

```bash
# Todos os sintéticos, datasets full (1M)
bash experiments/arte/run_mw_comparison.sh full

# Datasets específicos no modo reduzido
bash experiments/arte/run_mw_comparison.sh mini rbf_m agrawal_a
```

### Validação da checagem de direção do ADWIN

```bash
bash experiments/arte/run_adwin_direction.sh        # 1M instâncias
bash experiments/arte/run_adwin_direction.sh mini   # 50k instâncias
```

---

## 3. Executar experimentos GNN-ARTE

```bash
# Dispara 3 variantes (baseline, metagnn, metagnn_knn) para electricity, sea_g, outdoor
# Distribui automaticamente entre GPU 0 e GPU 1
bash experiments/gnn/run_experiments.sh
```

Variantes:

| Tag CSV | Argumento | Descrição |
|---|---|---|
| `baseline` | `--no_metagnn` | Votação majoritária simples |
| `metagnn` | `--graph_type full` | Meta-GNN com grafo completo |
| `metagnn_knn` | `--graph_type knn` | Meta-GNN com grafo KNN (k=5) |

Saída: `results/gnn/{dataset}_{tag}_s{seed}.csv`

---

## 4. Analisar resultados

```bash
# Comparação mw=5 vs mw=10 com referência MOA
python analysis/compare_mw.py --folder results/arte
python analysis/compare_mw.py --folder results/arte --metric kappa
python analysis/compare_mw.py --folder results/arte --full   # tabela linha a linha

# Sumário ARTE
python analysis/results_arte.py

# Sumário GNN-ARTE
python analysis/results_gnn.py
```

---

## 5. Datasets suportados

### Reais (ARFF em `DATASETS_PATH`)

| Chave | Arquivo | Instâncias |
|---|---|---|
| `electricity` | `elecNormNew.arff` | ~45k |
| `covtype` | `covtypeNorm.arff` | ~581k |
| `gassensor` | `gassensor.arff` | ~13k |
| `gmsc` | `GMSC.arff` | ~150k |
| `outdoor` | `outdoor.arff` | ~4k |
| `rialto` | `rialto.arff` | ~82k |
| `shuttle` | `shuttle.arff` | ~58k |

### Sintéticos gerados pelo MOA

| Chave | Descrição |
|---|---|
| `agrawal_a` / `agrawal_g` | Agrawal com drift abrupto / gradual |
| `led_a` / `led_g` | LED com drift abrupto / gradual |
| `sea_a` / `sea_g` | SEA com drift abrupto / gradual |
| `mixed_a` / `mixed_g` | Mixed com inversão de features |
| `rbf_f` | RBF Fast (deriva rápida) |
| `rbf_m` | RBF Moderate (deriva moderada) |

---

## 6. Notas técnicas

### ADWINChangeDetector e a checagem de direção

O `drift.ADWIN` do River sinaliza drift em **qualquer direção** (melhora ou piora de erro), o que causa uma cascata pós-reset: a nova árvore aprende rápido → erro cai → ADWIN interpreta melhora como drift → novo reset → espiral.

O `ADWINChangeDetector` em `src/arte/drift_detector.py` replica o comportamento do `ADWINChangeDetector.java` do MOA (linha 51): **só dispara quando o erro aumentou** (`estimation > prev_estimation`).

### Referências MOA (Tabela 14 do artigo)

Configuração: `seed=123456789`, `n_models=100`, `delta=1e-3`

| Dataset       | Acc (%)  | Kappa | Kappa_M | RAM MB | CPU (s)   |
|---------------|----------|-------|---------|--------|-----------|
| `agrawal_a`   | 79.50    | 0.589 | 0.565   | 12.633 | 12628.87  |
| `agrawal_g`   | 75.49    | 0.508 | 0.480   | 11.747 | 11640.75  |
| `airlines`    | 67.62    | 0.331 | 0.273   | 1.598  | 5295.78   |
| `covtype`     | 93.85    | 0.900 | 0.875   | 0.415  | 1192.48   |
| `electricity` | 90.31    | 0.801 | 0.772   | 0.092  | 62.84     |
| `gassensor`   | 94.41    | 0.932 | 0.927   | 0.302  | 242.82    |
| `gmsc`        | 93.55    | 0.140 | 0.025   | 2.166  | 653.20    |
| `keystroke`   | 93.69    | 0.916 | 0.865   | 0.069  | 3.33      |
| `led_a`       | 73.94    | 0.710 | 0.710   | 1.341  | 4355.00   |
| `led_g`       | 73.10    | 0.701 | 0.701   | 1.816  | 4590.23   |
| `outdoor`     | 74.30    | 0.736 | 0.736   | 0.045  | 8.71      |
| `ozone`       | 93.88    | 0.179 | 0.031   | 0.338  | 24.94     |
| `rbf_f`       | 79.86    | 0.736 | 0.712   | 0.257  | 2368.13   |
| `rbf_m`       | 87.71    | 0.840 | 0.824   | 1.389  | 5345.98   |
| `rialto`      | 81.05    | 0.789 | 0.789   | 0.044  | 470.76    |
| `sea_a`       | 89.71    | 0.783 | 0.743   | 3.918  | 7077.15   |
| `sea_g`       | 89.29    | 0.774 | 0.733   | 4.188  | 7064.23   |
| `shuttle`     | 99.77    | 0.994 | 0.989   | 0.191  | 82.43     |

### Reprodutibilidade

Todos os experimentos usam `seed=123456789` por padrão, replicando a configuração do artigo original.

---

## Dependências principais

```
river
deep-river
torch
torch-geometric  # opcional — GNN-ARTE usa fallback MLP se ausente
scipy
pandas
numpy
psutil
```
