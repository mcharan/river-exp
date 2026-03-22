# Nota Metodológica: Weight of Evidence (WOE) em Contexto de Aprendizado Online

## 1. Contexto

Durante o desenvolvimento do NeuralARTE, discutiu-se o uso de **Weight of Evidence (WOE)** como técnica de pré-processamento para codificação de atributos — especialmente atributos nominais com alta cardinalidade — antes da alimentação dos modelos neurais do ensemble. Esta nota documenta a análise da adequação do WOE ao contexto de *data streams* com *concept drift*.

---

## 2. Definição do WOE

O WOE é uma transformação supervisionada definida para problemas binários:

$$\text{WOE}_i = \ln\left(\frac{P(x = i \mid y = 1)}{P(x = i \mid y = 0)}\right) = \ln\left(\frac{n_{1i}/N_1}{n_{0i}/N_0}\right)$$

onde:
- $n_{1i}$ = número de instâncias da categoria $i$ pertencentes à classe positiva ($y=1$)
- $n_{0i}$ = número de instâncias da categoria $i$ pertencentes à classe negativa ($y=0$)
- $N_1$, $N_0$ = totais de cada classe no conjunto de treinamento

O WOE codifica cada categoria em um escalar que reflete sua relação log-linear com a variável alvo, sendo amplamente utilizado em modelos de *credit scoring*.

---

## 3. Vantagens em Contexto Estático

- **Encoding informativo**: captura a relação estatística entre categoria e alvo, superando One-Hot Encoding em modelos lineares.
- **Redução de dimensionalidade**: substitui $k$ colunas binárias (OHE) por uma única coluna numérica por variável.
- **Compatibilidade com redes neurais**: escalares WOE são diretamente consumíveis como *features* contínuas.
- **Tratamento natural de alta cardinalidade**: funciona bem mesmo com dezenas ou centenas de categorias distintas.

---

## 4. Limitações Críticas em Data Streams

### 4.1 Dependência de Estatísticas Globais

O cálculo clássico do WOE utiliza $P(x = i \mid y)$ estimada sobre **todo o conjunto de treinamento histórico**. Em *streams*, esse conjunto cresce indefinidamente e pode se tornar desatualizado após um *concept drift*, pois as probabilidades calculadas sobre instâncias antigas podem não refletir a distribuição atual.

### 4.2 Instabilidade sob Concept Drift

Após um drift, a relação entre uma categoria $i$ e a variável alvo pode se inverter. O WOE calculado antes do drift passará a codificar informação **incorreta ou enganosa**. Um modelo que consome esse WOE obsoleto pode ser prejudicado pela codificação, ao contrário de um modelo que consome a categoria como OHE ou embedding — cujos pesos podem ser ajustados pelo gradiente de forma independente por dimensão.

**Exemplo**: suponha que, antes de um drift, a categoria `A` seja fortemente associada à classe positiva (WOE > 0). Após o drift, ela passa a ser associada à classe negativa. O WOE permanecerá positivo até que novas observações suficientes sejam acumuladas, introduzindo ruído no sinal de entrada do modelo.

### 4.3 Problema com Janela Deslizante

Uma mitigação natural seria calcular o WOE sobre uma **janela deslizante** de tamanho $W$, atualizando as estimativas de $P(x = i \mid y)$ incrementalmente. Contudo, isso introduz dois novos problemas:

1. **Esparsidade**: para categorias raras, a janela pode não conter instâncias suficientes, levando a WOE extremo (divisão por zero ou $\pm\infty$) e necessitando suavização (add-k, suavização de Laplace).
2. **Hiperparâmetro sensível**: o tamanho $W$ da janela afeta diretamente a reatividade ao drift — janelas pequenas são ruidosas, janelas grandes são lentas.

### 4.4 Restrição a Problemas Binários

O WOE é intrinsecamente definido para dois classes ($y \in \{0, 1\}$). Em problemas multiclasse — como os presentes em vários *datasets* dos experimentos (e.g., `covtype` com 7 classes, `led` com 10 classes) — seria necessário uma extensão OvR (*One-vs-Rest*), produzindo $C$ valores WOE por atributo (um por classe), anulando a vantagem de compressão dimensional.

---

## 5. Comparação com Alternativas

| Técnica | Drift-resiliente | Multiclasse | Sem supervisão | Overhead |\
|---------|-----------------|-------------|----------------|----------|\
| One-Hot Encoding (OHE) | ✅ | ✅ | ✅ | Baixo |\
| Target Encoding (média da classe) | ⚠️ (janela) | ✅ | ❌ | Baixo |\
| **WOE** | ⚠️ (janela) | ❌ (OvR) | ❌ | Médio |\
| Embedding aprendido (nn.Embedding) | ✅ (gradiente) | ✅ | ❌ | Médio |\
| Label Encoding + normalização | ✅ | ✅ | ✅ | Muito baixo |\

Para redes neurais em *streams* com drift, **embeddings aprendidos** ou **OHE com normalização** são em geral mais robustos que o WOE, pois os pesos se adaptam pelo gradiente a cada atualização online.

---

## 6. Decisão Adotada

Nos experimentos do NeuralARTE, atributos nominais foram tratados com **NominalSplitterClassif** (no contexto das árvores clássicas do ARTE) e os modelos neurais do ensemble recebem as *features* codificadas numericamente pelo carregador universal (`get_dataset_universal`), que preserva a representação original dos atributos.

O WOE **não foi adotado** para os experimentos reportados, pela combinação de:
1. Fragilidade sob *concept drift* sem mecanismo explícito de janela
2. Complexidade adicional em problemas multiclasse
3. Ausência de evidência de ganho para redes neurais profundas (que aprendem internamente representações não-lineares das *features*)

---

## 7. Quando o WOE Poderia Ser Adequado

- Modelos **lineares ou logísticos** sobre streams estacionários ou com drift muito lento
- Problemas **binários** com atributos nominais de alta cardinalidade e distribuição estável
- Pipelines onde se deseja **interpretabilidade** dos coeficientes de cada categoria

---

## Referências

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
- Gama, J. (2010). *Knowledge Discovery from Data Streams*. Chapman & Hall/CRC.
- Losing, V., Hammer, B., & Wersing, H. (2018). *Incremental On-Line Learning: A Review and Comparison of State of the Art Algorithms*. Neurocomputing, 275, 1261–1274.
