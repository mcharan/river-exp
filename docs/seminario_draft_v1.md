# Esboço — Seminário DEEDS/WP1
> Draft v1 — Abril/2026

---

## 1. Contexto

- Fluxos contínuos de dados (*data streams*): chegada veloz, processamento de passada única, recursos limitados
- Mudança de conceito (*concept drift*): a distribuição dos dados muda ao longo do tempo, tornando modelos obsoletos
- Aprendizado por comitês (*ensembles*) é o estado da arte para fluxos: ARF, SRP, KUE — todos baseados em árvores de decisão (Hoeffding Trees)
- Interesse crescente em comitês **profundos** para fluxos (Dong & Japkowicz 2016; Luong et al. 2020), mas ainda pouco explorado
- Projeto DEEDS (CAPES/COFECUB): propõe investigar *deep ensembles* para mudança de conceito, anomalias e novidades — **WP1** foca especificamente em mudança de conceito

---

## 2. Problema

- Árvores de Hoeffding adaptam-se bem ao drift, mas têm **capacidade representacional limitada** para padrões complexos
- MLPs oferecem maior capacidade, mas apresentam desafios críticos em fluxos:
  - **Esquecimento catastrófico**: ao reiniciar após drift, toda representação aprendida é perdida
  - **Convergência lenta**: precisam de centenas/milhares de instâncias para re-aprender
  - **Sem protocolo estabelecido** para adaptação a drift em modo *online*
- **Pergunta central**: como construir um ensemble de MLPs que se adapte a mudanças de conceito sem perder o conhecimento acumulado?

---

## 3. Motivações

- Tarefa 1.1 do DEEDS: avaliar River (Python) e MOA (Java) como frameworks para implementação — escolha do framework adequado
- A arquitetura ARTE (*Adaptive Random Tree Ensemble*) é referência consagrada para drift com HTs: serve como **blueprint** para a versão neural
- Ao replicar o ARTE no River, identificamos uma **discrepância de implementação**: o ADWIN aplicado ao ARTE no MOA opera de forma direcional (só dispara quando o erro *aumenta*), comportamento ausente na implementação padrão do River — motivação para investigação e correção
- Aplicações reais demandam modelos capazes de lidar com deriva contínua: monitoramento industrial, análise financeira, sensores

---

## 4. Hipóteses

**H1** — Ensembles de MLPs com **diversidade de composição** (arquitetura + otimizador + taxa de aprendizado) formam classificadores eficazes para fluxos com drift.

**H2** — A detecção de drift **direcional** (apenas quando o erro *aumenta*) reduz falsos positivos e melhora o desempenho em ensembles neurais, comparado ao ADWIN padrão bidirecional.

**H3** — O **reset por membro individual** é superior ao reset do ensemble completo, pois isola a adaptação ao membro afetado sem degradar o ensemble inteiro.

**H4** — O **reset seletivo de camadas** (*soft reset*) — reinicializando apenas as camadas de decisão e preservando as camadas de extração de features — mitiga o esquecimento catastrófico e acelera a re-convergência pós-drift.

---

## 5. Objetivos

**Geral**: Desenvolver e validar métodos de comitês de modelos profundos para adaptação à mudança de conceito em fluxos de dados (DEEDS WP1).

**Específicos**:
1. Validar o framework River/Python como plataforma para *deep stream ensembles* (Tarefa 1.1)
2. Estabelecer uma implementação de referência do ARTE em Python e validar contra o MOA
3. Investigar estratégias de composição de ensemble neural para fluxos com drift
4. Propor e avaliar o reset seletivo por membro como mecanismo de adaptação explícita a drift

---

## 6. Atividades Desenvolvidas

| # | Atividade | Artefato |
|---|---|---|
| 1 | Implementação do ARTE em River + validação contra MOA (19 datasets) | `src/arte/` |
| 2 | Correção do ADWINChangeDetector: verificação direcional do erro | `src/arte/drift_detector.py` |
| 3 | Implementação do NeuralARTE (ARTELight): ensemble de MLPs com reset ensemble-level | `src/neural_arte/neural_arte.py` |
| 4 | Ablation de composições (abc, abc_proj, abc_cnn, current) em 20 datasets | `experiments/neural_arte/` |
| 5 | Implementação do ARTESoftResetNN: reset seletivo por membro individual | `src/neural_arte/arte_member_reset.py` |
| 6 | Scripts de execução em ondas e análise comparativa | `experiments/` + `analysis/` |

---

## 7. Resultados Preliminares

### 7.1 Validação do framework (Python ARTE vs MOA)
- Python melhor em **12/18** datasets, média **+1.21pp**
- Matches quase perfeitos em sea_g (0.00pp), sea_a (+0.03pp), electricity (-0.09pp) — confirma corretude da implementação
- `outdoor` +11.33pp, `agrawal_a/g` +4–5pp
- `airlines` -3.57pp — caso a investigar (possível diferença de versão do dataset)

### 7.2 Impacto do ADWINChangeDetector direcional
- Sem a verificação direcional, drifts falsos disparam resets desnecessários nos MLPs
- Com a correção, agrawal_a melhora ~4–6pp em relação à versão sem verificação

### 7.3 Ablation de composições (NeuralARTE)
- `abc_proj` melhor composição geral: **+0.50pp** vs `abc` (18 datasets)
- `abc_cnn`: **−0.51pp** vs `abc` — CNN prejudica datasets com muitos atributos nominais (OHE não preserva estrutura espacial)
- `current` polarizado: +4pp em keystroke/rialto, −8.6pp em agrawal_g

### 7.4 ARTESoftResetNN (reset seletivo)
- **+0.56pp** vs melhor composição NeuralARTE (abc_proj), vencendo em 11/18 datasets
- Maiores ganhos nos datasets com deriva: agrawal_a/g +4pp, gassensor +2pp
- Valida H3 e H4: reset individual + preservação de features reduz impacto do esquecimento catastrófico

### Em andamento
- Ablation adwin vs nodrift
- Ablation de composições no ARTESoftResetNN
- Rodadas completas (datasets full) para comparativo final

---

> **TODO para próxima versão**: adicionar figuras (curva de acurácia ao longo do tempo, tabela comparativa MOA vs Python, diagrama das arquiteturas NeuralARTE e ARTESoftResetNN).
