# GNN-ARTE: Meta-GNN para Agregação de Ensemble em Data Streams

Implementação da **Estratégia 2 (EnsembleGNN / Meta-GNN)** proposta na análise
`20260208-ANALISE_TRANSFORMERS_GNNS.md`. Substitui a votação majoritária do
NeuralARTE por uma Graph Attention Network (GAT) que aprende a agregar as
predições dos modelos base de forma adaptativa.

---

## Arquivos

| Arquivo | Descrição |
|---|---|
| `ensemble_gnn.py` | `EnsembleGNN` (GATConv), `EnsembleGNNFallback` (MLP), `MetaGNNAggregator` |
| `gnn_arte.py` | Loop principal do GNN-ARTE |
| `run_gnn_experiments.sh` | Disparo de experimentos (prova de conceito) |
| `results_gnn.py` | Análise comparativa dos CSVs gerados |

---

## Requisitos

### Ambiente base (já instalado no `deep-river-demo`)

- Python ≥ 3.9
- PyTorch ≥ 2.0 (com suporte CUDA)
- `deep-river`
- `river`
- `numpy`, `pandas`, `scipy`, `psutil`

### Dependências adicionais (necessário instalar)

```bash
# Ativa o ambiente
conda activate deep-river-demo

# Instala PyTorch Geometric e suas dependências C++
# Substitua cu118 pela versão CUDA do servidor (verifique com: nvcc --version)
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

pip install torch-geometric
```

> **Verificar a versão CUDA do servidor:**
> ```bash
> nvcc --version          # versão do compilador CUDA
> python -c "import torch; print(torch.version.cuda)"   # versão vista pelo PyTorch
> ```
> Ajuste `cu118` → `cu117`, `cu121`, etc. conforme necessário.

> **Fallback sem torch-geometric:** se a instalação não for possível, o código
> detecta automaticamente a ausência do pacote e usa `EnsembleGNNFallback`
> (MLP que recebe as predições concatenadas de todos os modelos). O pipeline
> completo funciona, mas sem as arestas do grafo real.

---

## Execução

### Prova de conceito (electricity, outdoor, sea_g — 3 variantes cada)

```bash
# No servidor, após git pull
chmod +x GNN/run_gnn_experiments.sh
bash GNN/run_gnn_experiments.sh
```

As 3 variantes por dataset são:
- `baseline` — votação majoritária (sem Meta-GNN)
- `metagnn` — Meta-GNN com grafo completo
- `metagnn_knn` — Meta-GNN com grafo k-NN (k=5, reconstruído por similaridade de predição)

### Execução manual

```bash
# Com Meta-GNN (grafo completo)
python GNN/gnn_arte.py --dataset electricity --n_models 30 --device cuda:0

# Baseline sem Meta-GNN (votação majoritária)
python GNN/gnn_arte.py --dataset electricity --n_models 30 --no_metagnn

# Com grafo k-NN
python GNN/gnn_arte.py --dataset electricity --graph_type knn
```

### Parâmetros disponíveis

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--dataset` | — | Nome do dataset (mesmo mapeamento do NeuralARTE) |
| `--seed` | 123456789 | Semente aleatória |
| `--n_models` | 30 | Tamanho do ensemble |
| `--lambda_val` | 6 | Parâmetro λ do Online Bagging (Poisson) |
| `--window` | 500 | Tamanho da janela deslizante |
| `--datasets_path` | (configurado no script) | Caminho para os ARFFs |
| `--device` | auto (cuda/cpu) | `cuda`, `cuda:0`, `cuda:1`, `cpu` |
| `--gnn_hidden` | 64 | Dimensão oculta das camadas GATConv |
| `--gnn_update` | 10 | Instâncias entre atualizações do Meta-GNN |
| `--gnn_heads` | 4 | Número de cabeças de atenção GAT |
| `--graph_type` | `full` | Topologia do grafo: `full` ou `knn` |
| `--no_metagnn` | False | Usa votação majoritária (baseline) |

---

## Análise de resultados

```bash
python GNN/results_gnn.py
```

Gera tabela comparativa e calcula ΔAcc / ΔLat entre `baseline` e `metagnn`.

---

## Estrutura dos CSVs gerados

Salvos em `results/gnn/<dataset>_<variante>_s<seed>.csv`:

| Coluna | Descrição |
|---|---|
| `Dataset` | Nome do dataset |
| `Instance` | Instância atual |
| `Accuracy` | Acurácia prequential acumulada |
| `Kappa` | Cohen's Kappa acumulado |
| `KappaM` | Kappa Temporal (Kappa-M) acumulado |
| `Drifts` | Total de drifts detectados |
| `Latencia_ms` | Latência de predição + agregação (ms) |
| `RAM_MB` | RSS do processo em MB |
| `Time` | Horário do log (HH:MM:SS) |

---

## Referências

- Veličković et al. (2018). *Graph Attention Networks*. ICLR.
- Fey & Lenssen (2019). *Fast Graph Representation Learning with PyTorch Geometric*. ICLR Workshop.
- Gomes et al. (2019). *Machine Learning for Streaming Data: State of the Art, Challenges, and Opportunities*. SIGKDD Explorations.
- Análise interna: `20260208-ANALISE_TRANSFORMERS_GNNS.md`
