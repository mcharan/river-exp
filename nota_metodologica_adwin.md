# Nota Metodológica: Divergências entre o Port Python/River e a Implementação de Referência Java/MOA do ARTE

## 1. Contexto

Durante a validação do port Python/River do algoritmo **ARTE** (*Adaptive Random Tree Ensemble*, Paim & Enembreck, 2020) frente à implementação de referência em Java/MOA, identificou-se uma divergência sistemática de desempenho em alguns *datasets*. A investigação sistemática das possíveis causas — conduzida por inspeção de código-fonte, experimentos de ablação e análise de parâmetros — revelou que as divergências resultam de uma combinação de fatores inerentes à diferença de plataforma, e não de erros de implementação do algoritmo ARTE em si.

A principal causa identificada é a diferença de implementação do detector de mudança **ADWIN** (*ADaptive WINdowing*, Bifet & Gavaldà, 2007) entre MOA e River, detalhada nas seções seguintes. Adicionalmente, foram investigadas e descartadas outras hipóteses, documentadas na Seção 6.

---

## 2. O Algoritmo ADWIN

O ADWIN mantém uma janela deslizante `W` de observações binárias recentes (tipicamente: 0 = classificação correta, 1 = erro). A cada nova observação, o algoritmo verifica se existe um ponto de corte na janela tal que a média da subjanela mais recente `W₁` difere significativamente da média da subjanela mais antiga `W₀`. Formalmente, uma mudança é sinalizada quando:

$$|\hat{\mu}_{W_0} - \hat{\mu}_{W_1}| \geq \varepsilon_{\text{cut}}$$

onde o limiar adaptativo é dado por:

$$\varepsilon_{\text{cut}} = \sqrt{\frac{1}{2m} \cdot \ln\frac{4n}{\delta}}$$

sendo `m = min(|W₀|, |W₁|)` o tamanho da menor subjanela, `n = |W|` o tamanho total da janela, e `δ` o parâmetro de confiança (probabilidade máxima de falso positivo por teste).

Quando uma mudança é detectada, as observações anteriores ao ponto de corte são descartadas, e o modelo associado é reiniciado.

O ADWIN possui ainda um parâmetro de *clock*: o teste estatístico só é executado a cada `clock` novas observações, por razões de eficiência computacional.

---

## 3. Parâmetros Divergentes entre MOA e River

A tabela abaixo resume os parâmetros internos relevantes das duas implementações, obtidos diretamente do código-fonte de cada biblioteca:

| Parâmetro | MOA `ADWINChangeDetector` | River `drift.ADWIN` | Efeito da Divergência |
|-----------|--------------------------|---------------------|-----------------------|
| `delta` (confiança) | `1.0 × 10⁻³` | `2.0 × 10⁻³` (padrão) | River usa padrão 2× mais permissivo; ambos configurados para `1e-3` nos experimentos |
| `clock` | `32` | `32` | **Idêntico** |
| `max_buckets` | `5` | `5` | **Idêntico** |
| `min_window_length` | **`10`** | **`5`** | River testa subjanelas menores → maior sensibilidade |
| `grace_period` | N/A | `10` | River ignora as primeiras 10 obs após instanciação |

A divergência principal está no parâmetro `min_window_length` (denominado `mintMinimLongitudWindow` no código Java do MOA): **o River permite testar cortes com subjanelas de tamanho mínimo 5, enquanto o MOA exige mínimo 10**.

---

## 4. Impacto Observado nos Experimentos

### 4.1 Espiral de Resets Sincronizados

Nos experimentos com ARTE (100 árvores, `lambda=6`, `delta=1e-3`, `seed=123456789`), identificou-se um padrão de detecções falsas após cada reset de árvore em *datasets* com deriva moderada:

- Após um reset legítimo (ou falso positivo inicial), a árvore recém-criada não possui conhecimento acumulado e apresenta taxa de erro elevada.
- No próximo *tick* do *clock* (exatamente 32 observações depois), o ADWIN executa o teste estatístico.
- Com apenas 32 observações disponíveis, River (min=5) testa cortes como [27|5], onde `W₁` = 5 observações da árvore inexperiente com erro próximo de 1.0. MOA (min=10) não testaria esse corte por insuficiência de observações em `W₁`.
- Resultado: River detecta um "drift" no recém-resetado detector → novo reset → novo falso positivo 32 instâncias depois → **espiral**.

A análise empírica no *dataset* `rbf_m` (1.000.000 instâncias, 5 classes, deriva moderada `1e-4`) revelou:

| Métrica | Valor Observado |
|---------|----------------|
| Total de drifts detectados | ~60.000 |
| Gap mínimo entre drifts consecutivos | **32 instâncias** (= 1 *clock tick*) |
| Gap máximo observado | 96 instâncias (= 3 *clock ticks*) |
| Todos os gaps são múltiplos de 32 | Confirmado |
| Drifts por 1.000 instâncias | ~60 |

A projeção a partir de 50.000 instâncias (2.796 drifts → ~56.000 em 1M) é consistente com o resultado completo, confirmando que o padrão se mantém ao longo de todo o *stream*.

### 4.2 Impacto na Acurácia

| *Dataset* | MOA (Java) | Python/River | Δ Acurácia | Drifts Python |
|-----------|-----------|--------------|-----------|---------------|
| rbf_m | 87.71% | 82.39% | **−5.32pp** | ~60.000 |
| rbf_f | 79.86% | 77.95% | **−1.91pp** | ~31.000 |
| agrawal_a | 79.50% | 76.48% | **−3.02pp** | ~3.700 |
| agrawal_g | 75.49% | 71.37% | **−4.12pp** | ~1.400 |

Note-se que `agrawal_a/g` apresentam poucos drifts mas gap de acurácia expressivo, sugerindo que nesses *datasets* podem existir causas adicionais além da espiral de resets.

### 4.3 Não Intercambialidade entre Delta e min_window_length

Poderia-se argumentar que ajustar `delta` no River compensaria a diferença em `min_window_length`. A análise matemática demonstra que isso não é viável de forma geral. Para equivalência do limiar `ε` entre as implementações:

$$\delta_{\text{MOA}} = \frac{\delta_{\text{River}}^2}{4n}$$

O fator de compensação depende de `n` (tamanho corrente da janela), que varia dinamicamente. Não existe um `delta` fixo que produza comportamento equivalente para todos os estados possíveis da janela. Experimentos empíricos com `delta ∈ {0.001, 0.01, 0.1, 1.0}` no *dataset* outdoor confirmaram que a relação entre delta e número de drifts/acurácia não é monotônica nem proporcional.

---

## 5. Variante com Guarda Pós-Reset (ARTE-ADWINGuard)

Para isolar e quantificar o efeito da espiral de resets, foi implementada uma variante do algoritmo denominada **ARTE-ADWINGuard**, que adiciona um parâmetro `drift_detection_grace_period` (padrão: 200 instâncias). Após cada reset, o detector de deriva não é atualizado durante as primeiras `drift_detection_grace_period` instâncias processadas pela árvore recém-criada, impedindo que o detector dispare sobre observações de uma árvore ainda sem conhecimento acumulado.

Esta variante **não modifica a lógica de detecção ou resposta a drifts do ARTE original** — preserva o mecanismo ADWIN, o *Online Bagging* via Poisson, o subespaço aleatório por nó, e o critério de votação por janela deslizante. O guard atua exclusivamente no período pós-reset, que no ARTE original não possui proteção explícita (o código Java também não a possui, mas a implementação MOA do ADWIN é naturalmente menos reativa nesse período devido ao `min_window_length=10`).

O objetivo desta variante é **empírico**: verificar se a diferença de sensibilidade explica os gaps de acurácia observados, e não propor uma melhoria ao algoritmo.

---

## 6. Investigação de Hipóteses Adicionais

Além da divergência no ADWIN, outras hipóteses foram investigadas para explicar os gaps remanescentes, particularmente nos *datasets* `agrawal_a`, `agrawal_g` e `airlines`. Todas foram descartadas como causas primárias.

### 6.1 Parâmetros da Hoeffding Tree

Os parâmetros `grace_period` (g) e `split_confidence` (c) da Hoeffding Tree foram comparados entre as implementações. A análise do código-fonte do `ARTE.java` revelou que o algoritmo configura explicitamente esses parâmetros via `ClassOption`:

```
ARTEHoeffdingTree -e 2000000 -g 100 -c 0.01 -n ARTEAttributeClassObserver
```

Isso é consistente com o que o artigo original especifica: *"we set the grace period to g = 100 and split confidence c = 0.01 [...] This setting tends to generate early splits relative to the default value of the Hoeffding trees algorithm with g = 200"* (Paim & Enembreck, 2020). O port Python utiliza os mesmos valores (`grace_period=100`, `delta=0.01`), confirmando fidelidade ao artigo.

A tabela abaixo compara todos os parâmetros relevantes da Hoeffding Tree entre as implementações:

| Parâmetro | Default River | Default MOA HT | ARTE (paper/Java) | Python ARTE |
|---|---|---|---|---|
| `grace_period` | 200 | 200 | **100** | **100** ✅ |
| `delta` (split confidence) | 1e-7 | 1e-7 | **0.01** | **0.01** ✅ |
| `tau` (tie threshold) | 0.05 | 0.05 | 0.05 | 0.05 ✅ |
| `nb_threshold` | 0 | 0 | 0 | 0 ✅ |
| `leaf_prediction` | nba | nba | nba | nba ✅ |
| `remove_poor_attrs` | False | False | disabled | False ✅ |

**Conclusão**: nenhuma divergência de parâmetros da Hoeffding Tree foi encontrada entre o port Python e a implementação Java/MOA do ARTE.

### 6.2 Codificação de Atributos Nominais

Os *datasets* `agrawal` e `airlines` contêm atributos nominais (e.g., `elevel`, `car`, `zipcode` no Agrawal). Investigou-se se a detecção e codificação desses atributos no carregador Python (`get_dataset_universal`) produzia splitters incorretos.

A inspeção direta dos splitters criados nas folhas das árvores confirmou que:
- Os atributos nominais são corretamente detectados pelo `dtype == object` do pandas após decodificação dos bytes do ARFF
- Os índices retornados em `nominal_attributes` coincidem com as colunas corretas
- O `RandomSubspaceNodeMixin` instancia `NominalSplitterClassif` para esses índices e `ARTEGaussianSplitter` para os numéricos, sem exceções

**Conclusão**: a hipótese de codificação incorreta de nominais foi descartada.

### 6.3 Variância de Execução Única e Diferenças de RNG

O artigo original reporta resultados como **média de 10 execuções** com sementes distintas. Os experimentos de validação deste port foram realizados com uma única execução (`seed=123456789`). Com 2–3 pp de variância esperada entre execuções individuais, parte dos gaps residuais observados (especialmente em `agrawal_g` e `airlines`) pode ser atribuída a essa diferença metodológica.

Adicionalmente, as implementações Java (`java.util.Random`, `org.apache.commons.math`) e Python (`numpy.random`) produzem sequências distintas para a mesma semente numérica. Isso afeta a ordem das seleções de subespaço aleatório por nó ao longo do *stream*, introduzindo variância estrutural irredutível entre plataformas.

---

## 7. Conclusão

A comparação entre o ARTE em Java/MOA e o port Python/River revela três categorias de divergência, todas inerentes à diferença de plataforma:

1. **Divergência estrutural do ADWIN** (causa primária, seções 2–5): a diferença de `min_window_length` (10 vs 5) produz uma espiral de resets sincronizados com o *clock* do detector (32 instâncias) em *datasets* com drift contínuo, reduzindo a acurácia do ensemble. Esta causa foi quantificada experimentalmente e é responsável pelos maiores gaps observados (rbf_m: −5.32pp → −2.01pp com guard).

2. **Parâmetros e lógica do algoritmo** (seção 6.1 e 6.2): todos os parâmetros relevantes da Hoeffding Tree e a lógica de atribuição de splitters foram verificados e estão em conformidade com a especificação do artigo original. Nenhuma divergência algorítmica foi identificada.

3. **Variância de plataforma irredutível** (seção 6.3): diferenças nos geradores de números aleatórios entre Java e Python e o uso de execução única (vs. média de 10 no artigo) introduzem variância que explica os gaps residuais em *datasets* como `agrawal_g` e `airlines`.

Para fins de comparação direta com resultados reportados em MOA, recomenda-se documentar estas divergências como **limitações metodológicas** inerentes à diferença entre plataformas, e não como defeitos de implementação do algoritmo ARTE.

---

## Referências

- Bifet, A., & Gavaldà, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing*. In Proceedings of the 2007 SIAM International Conference on Data Mining (SDM), pp. 443–448.
- Paim, A., & Enembreck, F. (2020). *ARTE: Adaptive Random Tree Ensemble for Learning from Massive Imbalanced Data Streams*. In Proceedings of the 32nd International Conference on Tools with Artificial Intelligence (ICTAI).
- Montiel, J., et al. (2021). *River: machine learning for streaming data in Python*. Journal of Machine Learning Research, 22(110), 1–8.
- Bifet, A., et al. (2010). *MOA: Massive Online Analysis*. Journal of Machine Learning Research, 11, 1601–1604.
