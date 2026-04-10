# CLAUDE.md — Contexto do Projeto river-exp

Este arquivo serve de ponto de entrada para o Claude Code continuar o trabalho
de desenvolvimento e experimentação do projeto DEEDS/WP1 a partir de qualquer sessão.

---

## Visão Geral do Projeto

Implementação e avaliação de **comitês de modelos profundos para fluxos de dados com mudança de conceito**,
no contexto do projeto DEEDS (CAPES/COFECUB), WP1 — Mudança de Conceito.

O blueprint algorítmico é o **ARTE** (*Adaptive Random Tree Ensemble*), originalmente implementado no MOA
com Hoeffding Trees. O objetivo é portar e estender o conceito para redes neurais (MLPs) no River (Python).

**Orientador**: Prof. Dr. Jean Paul Barddal (PUCPR)
**Referência do projeto**: `docs/JeanPaulBarddal-PROPOSTA_CAPES.pdf`
**Esboço do seminário**: `docs/seminario_draft_v1.md`

---

## Estrutura do Repositório

```
river-exp/
├── src/
│   ├── arte/               # ARTE original com Hoeffding Trees
│   │   ├── ensemble.py     # ARTEEnsemble — implementação principal
│   │   ├── drift_detector.py  # ADWINChangeDetector (direcional — ver nota abaixo)
│   │   ├── tree.py
│   │   ├── nodes.py
│   │   └── splitter.py
│   ├── neural_arte/
│   │   ├── neural_arte.py       # NeuralARTE (ARTELight) — reset ensemble-level
│   │   └── arte_member_reset.py # ARTESoftResetNN + ARTESubspaceNN — reset por membro
│   ├── gnn/
│   │   └── gnn_arte.py          # GNN-ARTE (experimento, resultados fracos)
│   └── shared/
│       └── metrics.py           # KappaM (métrica custom)
├── experiments/
│   ├── arte/
│   │   └── run_all.sh           # Ondas: --wave N --mw M --gpu G
│   ├── neural_arte/
│   │   ├── run_all.sh           # Ondas: --wave N --composition X --gpu G --no_drift
│   │   ├── run_ablation.sh      # Ablation de composições (full datasets)
│   │   ├── run_ablation_mini.sh # Ablation em mini datasets (2 eixos: comp e n_models)
│   │   └── run_arte_member.sh   # Ondas para ARTESoftResetNN/ARTESubspaceNN
│   └── neural_arte/
│       ├── run_arte_member.py   # CLI: --arch soft_reset|subspace --dataset ...
│       └── neural_arte.py       # CLI: --dataset ... --composition ... --no_drift
├── analysis/
│   ├── results_arte.py          # Tabela de resultados do ARTE (com coluna RODADA)
│   ├── results_neural.py        # Tabela de resultados do NeuralARTE
│   ├── results_soft_reset.py    # Tabela de resultados do ARTESoftResetNN
│   ├── compare_ablation.py      # Pivot: composição x dataset (NeuralARTE)
│   ├── compare_moa_python.py    # Comparativo ARTE Python vs MOA lado a lado
│   └── compare_mw.py            # Comparativo mw=5 vs mw=10
├── results/
│   ├── arte/                    # CSVs do ARTE Python
│   │   # Padrão: ARTE_CPU_{dataset}_mw{mw}_s{seed}_{YYYYMMDD_HHMMSS}.csv
│   └── neural/                  # CSVs do NeuralARTE e SoftReset
│       # NeuralARTE: NeuralARTE_{dataset}_{composition}_{drift_tag}_s{seed}_{ts}.csv
│       # SoftReset:  NeuralARTE_{dataset}_soft_reset_{comp}_rl{n}_{drift}_s{seed}_{ts}.csv
├── results/logs/                # Logs das sessões screen
└── docs/
    └── seminario_draft_v1.md    # Esboço do seminário (v1, abril/2026)
```

---

## Variáveis de Ambiente Necessárias

Sempre exportar antes de chamar os scripts:

```bash
export DATASETS_PATH=/caminho/para/datasets/mini   # ARFFs mini (sintéticos)
export DATASETS_PATH_REAL=/caminho/para/datasets/full  # ARFFs full (todos os datasets)
export PYTHON=python3   # ou caminho do venv
```

O `DATASETS_PATH_REAL` é necessário como fallback para datasets reais (electricity, outdoor,
airlines, etc.) que não existem na pasta mini.

---

## Decisões Técnicas Importantes

### ADWINChangeDetector — verificação direcional
`src/arte/drift_detector.py` implementa um wrapper sobre `river.drift.ADWIN` que **só dispara
drift quando o erro aumenta** (`estimation > prev_estimation`). Isso replica o comportamento
do MOA e evita falsos positivos em períodos de melhora — crítico para MLPs, onde um reset
desnecessário causa esquecimento catastrófico.

Parâmetros de referência:
- `delta=0.001` (padrão MOA)
- `min_window_length=10` (mw=10, equivalente ao MOA; padrão River é 5)

### ARTE Python vs MOA — resultado de validação
Com `mw=10`, seed=123456789, n_models=100:
- Python melhor em **12/18** datasets, média **+1.21pp**
- sea_g: 0.00pp (match perfeito), electricity: -0.09pp
- Referência MOA disponível em: `results_moa_baseline/` (no host da faculdade)

### Composições do ensemble (NeuralARTE)
Definidas em `src/neural_arte/neural_arte.py` → `COMPOSITIONS`:
- `abc`: Fast(SGD)+Deep(Adam)+Mid(Adam) — baseline
- `abc_proj`: abc com versões com projeção ortogonal — melhor geral (+0.50pp vs abc)
- `abc_cnn`: abc + tier CNN — pior que abc em média (-0.51pp); CNN prejudica datasets com OHE
- `current`: MLP_Simple+MLP_CNN+MLP_Proj — composição original, polarizada
- `abc_extended`: abc + Wide + RMSprop

### ARTESoftResetNN — reset seletivo por membro
`src/neural_arte/arte_member_reset.py`:
- Cada membro tem seu próprio `ADWINChangeDetector`
- No drift: reinicializa apenas as últimas `n_reset_layers` camadas lineares (kaiming_uniform)
  e recria o optimizer (limpa momentum)
- Preserva camadas iniciais (extração de features)
- **Atenção**: para MLP_Fast (layers=[64]) com n_reset_layers=1, o soft reset equivale a ~50%
  da rede — considerar n_reset_layers=2 no ablation
- Resultado atual (abc, rl=1): **+0.56pp** vs NeuralARTE abc_proj em 18 datasets

### ARTESubspaceNN — random subspace (descartado)
Também em `arte_member_reset.py`. Testado no electricity:
- delta=0.001: 85% acc, ~440 drifts — drifts excessivos, MLP não converge
- delta=0.0001: 85.25% acc, 292 drifts — não melhorou significativamente
- **Conclusão**: Random subspace não funciona bem com MLPs (convergência lenta pós-reset).
  Mantido no código mas não priorizado para experimentos.

---

## Estado Atual dos Experimentos (abril/2026)

### Concluídos
- [x] ARTE Python validado vs MOA (18/19 datasets — falta led_a)
- [x] NeuralARTE ablation de composições (abc, abc_proj, abc_cnn, current) em 20 mini datasets
- [x] ARTESoftResetNN rodada inicial (abc, rl=1, adwin) em 20 datasets full

### Em execução (host da faculdade)
- [ ] NeuralARTE nodrift ablation — `run_all.sh --no_drift --wave 3 --gpu X`
      (necessário para popular `tabela_drift_vs_nodrift` no `compare_ablation.py`)
- [ ] ARTESoftResetNN ablation de composições (abc_proj, current, abc_cnn)
- [ ] ARTE Python led_a (mw=10) para fechar comparativo MOA

### Pendentes (próximas rodadas no hardware da faculdade)
- [ ] ARTESoftResetNN com n_reset_layers=2 (investigar impacto em MLPs rasos)
- [ ] ARTESoftResetNN rodadas full com melhor composição do ablation
- [ ] Comparativo final: ARTE HT vs NeuralARTE vs ARTESoftResetNN vs MOA

---

## Comandos de Referência

### Executar ARTE em ondas
```bash
bash experiments/arte/run_all.sh --wave 3 --mw 10 --gpu 0
```

### Executar NeuralARTE em ondas
```bash
bash experiments/neural_arte/run_all.sh --wave 3 --composition abc_proj --gpu 1
bash experiments/neural_arte/run_all.sh --wave 3 --composition abc --no_drift --gpu 0
```

### Executar ARTESoftResetNN
```bash
bash experiments/neural_arte/run_arte_member.sh --arch soft_reset --gpu 0 --wave 3
```

### Análise de resultados
```bash
# ARTE vs MOA
python analysis/compare_moa_python.py \
    --moa_folder /caminho/results_moa_baseline \
    --python_folder results/arte --mw 10

# Ablation composições (NeuralARTE)
python analysis/compare_ablation.py

# SoftReset
python analysis/results_soft_reset.py
python analysis/results_soft_reset.py --full  # tabela linha a linha
python analysis/results_soft_reset.py --metric kappa  # pivot por kappa
```

---

## Próximas Decisões de Pesquisa

1. **Ablation de composições no ARTESoftResetNN**: qual composição funciona melhor com reset seletivo?
   Hipótese: `abc_proj` deve manter vantagem; `abc_cnn` pode se beneficiar mais do soft reset
   (CNN preserva mais representação nas camadas iniciais).

2. **n_reset_layers**: testar rl=2 especialmente para membros MLP_Fast (apenas 2 layers lineares).
   Com rl=1, Fast já perde ~50% do conhecimento — rl=2 equivale a reset total para Fast.

3. **Composição híbrida**: combinar tiers do `abc_proj` com tier CNN nos datasets onde CNN mostrou
   vantagem (keystroke, led_a/g, rialto). Criar nova composição `abc_proj_cnn`.

4. **Adwin vs nodrift**: medir quanto do ganho do SoftReset vem do detector e quanto da
   adaptação natural dos MLPs ao longo do tempo.

---

## Referências Bibliográficas Relevantes

- Bifet & Gavalda (2007) — ADWIN: Learning from Time-Changing Data with Adaptive Windowing
  **Leitura pendente**: verificar se o paper descreve comportamento direcional ou bidirecional
- Gomes et al. (2017) — ARF: Adaptive Random Forests for Evolving Data Stream Classification
- Gomes et al. (2019) — SRP: Streaming Random Patches
- Dong & Japkowicz (2016) — Threaded ensembles of supervised/unsupervised NNs for stream learning
- Luong et al. (2020) — Streaming Active Deep Forest for evolving data stream classification
