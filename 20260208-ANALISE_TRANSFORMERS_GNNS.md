# ANÁLISE: TRANSFORMERS E GNNs PARA MACHINE LEARNING EM STREAMING
## Potencial, Viabilidade e Estratégias de Implementação

---

## 1. VISÃO GERAL: ESTADO ATUAL vs ARQUITETURAS AVANÇADAS

### 1.1 Contexto Atual do Projeto

**Baseline Atual (Neural ARTE)**:
- MLPs com 2-3 camadas: [64, 32] ou [128, 64]
- Parâmetros: ~10k-50k por modelo
- Latência: 1-3ms por instância (com batching)
- Ensemble: 50-100 modelos
- Memória GPU: ~200-500MB
- Throughput: 300-500 inst/s

### 1.2 Arquiteturas Avançadas Propostas

| Arquitetura | Parâmetros | Latência Esperada | Memória | Adequação |
|-------------|------------|-------------------|---------|-----------|
| **Transformer** | 100k-1M+ | 5-20ms | 1-3GB | ⚠️ Condicional |
| **GNN** | 50k-500k | 3-10ms | 500MB-2GB | ✅ Promissor |
| **MLP Atual** | 10k-50k | 1-3ms | 200-500MB | ✅ Baseline |

---

## 2. TRANSFORMERS PARA STREAMING: ANÁLISE CRÍTICA

### 2.1 Conceito e Mecânica

**Transformer** = Attention Mechanism + Feed-Forward Networks

```
Input Sequence → Self-Attention → Add & Norm → FFN → Add & Norm → Output
```

**Componentes Principais**:
1. **Self-Attention**: Q, K, V matrices (complexidade O(n²))
2. **Multi-Head Attention**: Múltiplas perspectivas paralelas
3. **Positional Encoding**: Informação de ordem temporal
4. **Feed-Forward**: MLP com ativação não-linear

### 2.2 Desafios para Data Streams Tabulares

#### 2.2.1 Problema Fundamental: SEQUÊNCIA vs INSTÂNCIA

**Transformers são projetados para sequências**:
- NLP: "The cat sat on the mat" → 6 tokens
- Visão: Patches de imagem → 196 patches (14×14)
- Áudio: Frames temporais → 1000+ frames

**Data Streams Tabulares**:
- Elec2: 1 instância = 8 features (ÚNICO vetor)
- Não há sequência inerente nas features
- Cada instância é independente (não há dependência temporal entre features)

**Implicação**: Transformers perdem sua principal vantagem (modelar relações sequenciais).

#### 2.2.2 Overhead Computacional

**Complexidade Self-Attention**: O(n² × d)
- n = comprimento da sequência
- d = dimensão do embedding

Para features tabulares:
- n = 8 (Elec2) ou 17 (após OHE) → muito pequeno para beneficiar atenção
- d = 64 (típico) → overhead de multiplicação matricial

**Comparação com MLP**:
```python
# MLP: O(d₁ × d₂)
forward_mlp = x @ W1 + b1  # (batch, 8) @ (8, 64) = O(8×64)

# Transformer: O(n² × d) + O(n × d²)
Q, K, V = x @ Wq, x @ Wk, x @ Wv  # 3 × O(8×64)
attention = softmax(Q @ K.T / sqrt(d)) @ V  # O(8² × 64) + O(8 × 64²)
```

Para n=8, o Transformer é ~5-10x mais lento que MLP sem ganho expressivo.

### 2.3 Cenários Onde Transformers SERIAM Úteis

#### Caso 1: Sliding Window sobre Stream

**Ideia**: Tratar últimas N instâncias como sequência temporal

```python
class TemporalTransformer(nn.Module):
    """
    Transforma janela deslizante de instâncias em sequência para Transformer
    """
    def __init__(self, n_features, n_classes, window_size=32, 
                 d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.window_size = window_size
        
        # Embedding de features para dimensão do Transformer
        self.feature_embedding = nn.Linear(n_features, d_model)
        
        # Positional Encoding (tempo no stream)
        self.pos_encoding = nn.Parameter(
            torch.randn(window_size, d_model)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classificador final
        self.classifier = nn.Linear(d_model, n_classes)
    
    def forward(self, x_window):
        """
        Args:
            x_window: (batch, window_size, n_features)
        Returns:
            logits: (batch, n_classes)
        """
        batch_size = x_window.shape[0]
        
        # Embedding: (batch, window, features) → (batch, window, d_model)
        embedded = self.feature_embedding(x_window)
        
        # Adiciona positional encoding
        embedded = embedded + self.pos_encoding.unsqueeze(0)
        
        # Self-attention sobre a janela temporal
        # Captura: "a instância 5 atrás influencia a predição atual?"
        context = self.transformer(embedded)
        
        # Pooling temporal: usa último timestep (mais recente)
        last_context = context[:, -1, :]
        
        # Classificação
        return self.classifier(last_context)

# Uso em streaming
class StreamingTransformerWrapper:
    def __init__(self, model, window_size=32):
        self.model = model
        self.window_size = window_size
        self.buffer = []  # Janela deslizante
    
    def predict_one(self, x):
        # Adiciona ao buffer
        self.buffer.append(x)
        
        # Mantém apenas últimas window_size instâncias
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Se buffer incompleto, padding
        if len(self.buffer) < self.window_size:
            padded = np.zeros((self.window_size, len(x)))
            padded[-len(self.buffer):] = self.buffer
        else:
            padded = np.array(self.buffer)
        
        # Inferência
        with torch.no_grad():
            window_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
            logits = self.model(window_tensor)
            return torch.argmax(logits, dim=-1).item()
    
    def learn_one(self, x, y):
        # Similar ao predict_one para manter buffer
        self.buffer.append(x)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Treina com contexto temporal
        if len(self.buffer) == self.window_size:
            window_tensor = torch.tensor(self.buffer, dtype=torch.float32).unsqueeze(0)
            target = torch.tensor([y], dtype=torch.long)
            
            # Backprop
            self.model.train()
            loss = F.cross_entropy(self.model(window_tensor), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

**Vantagens deste Approach**:
- ✅ Captura dependências temporais: "Se últimas 5 instâncias foram classe A, próxima tende a ser B"
- ✅ Detecta padrões de drift: Transformer "vê" mudanças graduais na distribuição
- ✅ Contexto rico para concept drift

**Desvantagens**:
- ❌ Latência: 32×8 features → atenção sobre 256 valores → ~10-15ms
- ❌ Memória: Buffer de 32 instâncias × 100 modelos = 3200 instâncias em RAM
- ❌ Cold Start: Primeiras 32 instâncias têm predições ruins (buffer incompleto)

**Análise de Viabilidade**: ⚠️ **CONDICIONAL**
- Datasets com autocorrelação temporal forte (ex: séries temporais): ✅ Vale a pena
- Datasets IID (ex: Agrawal sintético): ❌ Overhead sem benefício

#### Caso 2: Feature Attention (Cross-Feature)

**Ideia**: Usar atenção para aprender importância relativa entre features

```python
class FeatureAttentionTransformer(nn.Module):
    """
    Aplica self-attention sobre features (não temporal)
    """
    def __init__(self, n_features, n_classes, d_model=64):
        super().__init__()
        
        # Cada feature vira um "token"
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        
        # Transformer para aprender relações entre features
        # Ex: "feature 2 é importante quando feature 5 está alta"
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=4,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * n_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, n_features)
        Returns:
            logits: (batch, n_classes)
        """
        batch_size = x.shape[0]
        
        # Embedding individual por feature: (batch, n_features, d_model)
        embedded = torch.stack([
            emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeddings)
        ], dim=1)
        
        # Self-attention: aprende "feature 3 depende de feature 7"
        attended, _ = self.attention(embedded, embedded, embedded)
        
        # Flatten e classificar
        flattened = attended.reshape(batch_size, -1)
        return self.classifier(flattened)
```

**Quando isso é útil**:
- Datasets com interações complexas entre features
- Exemplo: Fraude (relação entre "valor transação" × "hora do dia" × "localização")

**Problema**: Para tabular simples (Elec2), MLP já aprende essas relações via camadas ocultas.

### 2.4 Estimativa de Performance: Transformer vs MLP

**Experimento Hipotético** (Elec2, 100 modelos, batch=32):

| Métrica | MLP [64,32] | Transformer (window=32) | Ratio |
|---------|-------------|-------------------------|-------|
| Parâmetros/modelo | 15k | 250k | 16.7x |
| VRAM Total | 300MB | 5GB | 16.7x |
| Latência (forward) | 0.5ms | 8ms | 16x |
| Latência (backward) | 1.5ms | 20ms | 13.3x |
| **Throughput** | **400 inst/s** | **25 inst/s** | **0.06x** |

**Conclusão**: Transformers reduziriam throughput em ~16x sem evidência clara de ganho em acurácia para tabulares.

### 2.5 Recomendação para Transformers

**NÃO USAR** como base learner principal, EXCETO:
1. Datasets com forte autocorrelação temporal (séries financeiras, sensor IoT)
2. Como modelo de "ensemble de 2º nível" (meta-learner):
   - Base: 100 MLPs fazem predições
   - Transformer: analisa últimas 32 predições do ensemble para fazer decisão final
   - Captura dinâmica do ensemble ao longo do tempo

**Implementação Sugerida para Meta-Learning**:
```python
class TransformerMetaLearner:
    """
    Usa Transformer para agregar predições históricas do ensemble
    """
    def __init__(self, base_ensemble, window_size=32):
        self.base_ensemble = base_ensemble
        self.prediction_history = []
        self.window_size = window_size
        
        # Transformer que analisa últimas 32 predições
        self.meta_model = TemporalTransformer(
            n_features=len(base_ensemble.models),  # Cada modelo = 1 feature
            n_classes=2,
            window_size=window_size
        )
    
    def predict_one(self, x):
        # 1. Todas as predições base
        base_preds = [m.predict_proba_one(x) for m in self.base_ensemble.models]
        
        # 2. Adiciona ao histórico
        self.prediction_history.append(base_preds)
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
        
        # 3. Transformer analisa contexto temporal das predições
        if len(self.prediction_history) == self.window_size:
            context = torch.tensor(self.prediction_history)
            final_pred = self.meta_model(context.unsqueeze(0))
            return final_pred
        else:
            # Fallback: votação majoritária enquanto buffer enche
            return max(base_preds, key=base_preds.get)
```

---

## 3. GRAPH NEURAL NETWORKS (GNNs): ANÁLISE PROMISSORA

### 3.1 Por que GNNs são Mais Adequados que Transformers?

**Resposta**: GNNs modelam **relações estruturadas**, não sequenciais.

Para dados tabulares:
- Cada instância pode ser vista como **grafo de features**
- Edges conectam features correlacionadas
- Message passing captura interações locais (mais eficiente que atenção global)

### 3.2 Três Estratégias de Aplicação de GNNs

#### Estratégia 1: Feature Interaction Graph (Dentro de Instância)

**Ideia**: Cada instância vira um grafo onde nós = features

```python
class FeatureGraphNN(nn.Module):
    """
    Constrói grafo de features e aplica GNN para capturar interações
    """
    def __init__(self, n_features, n_classes, hidden_dim=64):
        super().__init__()
        self.n_features = n_features
        
        # Feature embeddings (cada feature vira um nó)
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(n_features)
        ])
        
        # Graph Convolutional Network
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Readout (agregação de nós para predição)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
        
        # Estrutura do grafo (pode ser aprendida ou fixa)
        self.edge_index = self._build_graph_structure()
    
    def _build_graph_structure(self):
        """
        Estratégias para conectar features:
        1. FULLY CONNECTED: Todas as features conectadas
        2. CORRELATION-BASED: Apenas features correlacionadas
        3. LEARNED: Aprende conexões durante treino
        """
        # Opção 1: Fully Connected (baseline)
        edges = []
        for i in range(self.n_features):
            for j in range(self.n_features):
                if i != j:
                    edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def forward(self, x, batch_index=None):
        """
        Args:
            x: (batch, n_features)
        Returns:
            logits: (batch, n_classes)
        """
        batch_size = x.shape[0]
        
        # Embedding de cada feature
        node_features = torch.stack([
            emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeddings)
        ], dim=1)  # (batch, n_features, hidden_dim)
        
        # Flatten para processar todas as instâncias em batch
        node_features = node_features.reshape(-1, node_features.shape[-1])
        
        # Constrói edge_index para batch (replica grafo para cada instância)
        batch_edge_index = []
        for b in range(batch_size):
            offset = b * self.n_features
            batch_edge_index.append(self.edge_index + offset)
        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        
        # Message Passing (GNN layers)
        h = F.relu(self.conv1(node_features, batch_edge_index))
        h = F.relu(self.conv2(h, batch_edge_index))
        
        # Reshape de volta: (batch * n_features, hidden) → (batch, n_features, hidden)
        h = h.reshape(batch_size, self.n_features, -1)
        
        # Readout: agrega informação de todos os nós
        h_pooled = h.reshape(batch_size, -1)
        
        return self.classifier(h_pooled)


# Wrapper para River
class GNNClassifier(base.Classifier):
    def __init__(self, n_features, n_classes, device='cuda'):
        self.model = FeatureGraphNN(n_features, n_classes).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
    
    def learn_one(self, x, y):
        # Converte dict para tensor
        x_tensor = torch.tensor(list(x.values()), dtype=torch.float32).unsqueeze(0).to(self.device)
        y_tensor = torch.tensor([y], dtype=torch.long).to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(x_tensor)
        loss = self.loss_fn(logits, y_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        return self
    
    def predict_proba_one(self, x):
        x_tensor = torch.tensor(list(x.values()), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        return {i: float(p) for i, p in enumerate(probs)}
```

**Complexidade**:
- Message Passing: O(E × d) onde E = número de arestas
- Para grafo completo: E = n_features × (n_features - 1) = 8×7 = 56 (Elec2)
- Comparado a MLP: similar ou levemente superior

**Vantagens sobre MLP**:
- ✅ Captura relações feature-to-feature explicitamente
- ✅ Pode adaptar estrutura do grafo durante treino
- ✅ Generalizável: adicionar feature = adicionar nó (vs MLP que requer retreino completo)

**Estimativa de Performance**:
- Parâmetros: ~50k-100k (similar a MLP média)
- Latência: 2-4ms (1.5-2x MLP)
- VRAM: ~500MB para 100 modelos
- **Throughput esperado**: 200-300 inst/s (vs 400 MLP)

**Recomendação**: ✅ **VALE TESTAR** - overhead aceitável com potencial ganho em expressividade.

#### Estratégia 2: Ensemble-as-Graph (Meta-GNN)

**Ideia**: Tratar ensemble como grafo onde cada modelo é um nó

```python
class EnsembleGNN(nn.Module):
    """
    GNN que opera sobre o grafo de modelos do ensemble
    """
    def __init__(self, n_models, n_classes, hidden_dim=64):
        super().__init__()
        
        # Cada nó (modelo) tem features: [acurácia recente, tipo modelo, predição atual]
        self.node_encoder = nn.Linear(3 + n_classes, hidden_dim)
        
        # GNN para message passing entre modelos
        from torch_geometric.nn import GATConv  # Graph Attention
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # Agregação final
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, model_features, model_predictions, edge_index):
        """
        Args:
            model_features: (n_models, 3) - [accuracy, model_type_id, drift_count]
            model_predictions: (n_models, n_classes) - probabilidades de cada modelo
            edge_index: (2, E) - conexões entre modelos
        Returns:
            final_prediction: (n_classes,) - predição agregada
        """
        # Concatena features + predições
        node_features = torch.cat([model_features, model_predictions], dim=-1)
        
        # Encoding
        h = self.node_encoder(node_features)
        
        # Message Passing: modelos "conversam" entre si
        # Ex: modelo com alta acurácia influencia vizinhos
        h = F.relu(self.gat1(h, edge_index))
        h = F.relu(self.gat2(h, edge_index))
        
        # Global pooling: agrega todos os modelos
        h_pooled = torch.mean(h, dim=0)  # ou torch.max
        
        return self.readout(h_pooled)
    
    def _build_ensemble_graph(self, model_types):
        """
        Conecta modelos por similaridade:
        - Mesmo tipo (MLP_Simple ↔ MLP_Simple)
        - Mesmo otimizador
        - Acurácia próxima
        """
        edges = []
        for i in range(len(model_types)):
            for j in range(i+1, len(model_types)):
                # Conecta se mesmo tipo
                if model_types[i] == model_types[j]:
                    edges.append([i, j])
                    edges.append([j, i])  # Bidirecional
        
        return torch.tensor(edges, dtype=torch.long).t()
```

**Uso no ARTELight**:
```python
class ARTELightGNN(ARTELight):
    def __init__(self, models, model_types, drift_detector, **kwargs):
        super().__init__(models, model_types, drift_detector, **kwargs)
        
        # Meta-GNN para agregação inteligente
        self.meta_gnn = EnsembleGNN(
            n_models=len(models),
            n_classes=2  # ou autodetectar
        )
        
        # Grafo de conexões entre modelos
        self.edge_index = self._build_ensemble_graph(model_types)
    
    def predict_proba_one(self, x):
        # 1. Predições de todos os modelos
        model_preds = []
        model_features = []
        
        for i, m in enumerate(self.models):
            proba = m.predict_proba_one(x)
            model_preds.append([proba.get(0, 0), proba.get(1, 0)])
            
            # Features do modelo: [acc, type_id, drift_count]
            model_features.append([
                self._acc_windows[i].get(),
                float(self.model_types[i] == 'MLP_Proj'),  # 0 ou 1
                float(self._detectors[i].drift_detected)
            ])
        
        # Converte para tensors
        preds_tensor = torch.tensor(model_preds, dtype=torch.float32)
        feats_tensor = torch.tensor(model_features, dtype=torch.float32)
        
        # 2. GNN agrega informação do ensemble
        final_proba = self.meta_gnn(feats_tensor, preds_tensor, self.edge_index)
        
        # 3. Retorna como dict River
        proba_np = F.softmax(final_proba, dim=-1).detach().cpu().numpy()
        return {i: float(p) for i, p in enumerate(proba_np)}
```

**Vantagens**:
- ✅ Aprende dinâmica do ensemble (quais modelos confiar em cada contexto)
- ✅ Detecta "expertise local": modelo especialista em certa região do espaço
- ✅ Overhead baixo: GNN roda 1 vez (vs 100 forward passes dos modelos)

**Complexidade Adicional**:
- Forward GNN: ~1ms (apenas 1 grafo de 100 nós)
- VRAM adicional: ~50MB
- **Impacto total em throughput**: ~5-10% slower (de 400→360 inst/s)

**Recomendação**: ✅✅ **ALTAMENTE PROMISSOR** - overhead mínimo com potencial grande ganho.

#### Estratégia 3: Temporal GNN (Stream-as-Graph)

**Ideia**: Modelar stream como grafo temporal onde instâncias vizinhas conectam-se

```python
class TemporalGNN(nn.Module):
    """
    Grafo temporal: nós = instâncias, arestas = proximidade temporal
    """
    def __init__(self, n_features, n_classes, window_size=32):
        super().__init__()
        self.window_size = window_size
        
        # Encoder de instância → nó
        self.node_encoder = nn.Linear(n_features, 64)
        
        # Temporal GNN
        from torch_geometric.nn import GATConv
        self.tgnn = GATConv(64, 64, heads=4, concat=False)
        
        self.classifier = nn.Linear(64, n_classes)
    
    def forward(self, x_window):
        """
        Args:
            x_window: (window_size, n_features) - últimas N instâncias
        Returns:
            logits: (n_classes,) - predição para instância mais recente
        """
        # Encoding de cada instância
        node_features = self.node_encoder(x_window)  # (window_size, 64)
        
        # Constrói grafo temporal: conecta instâncias consecutivas
        edge_index = []
        for i in range(self.window_size - 1):
            edge_index.append([i, i+1])  # t → t+1
            edge_index.append([i+1, i])  # bidirecional
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Message Passing temporal
        h = F.relu(self.tgnn(node_features, edge_index))
        
        # Predição usa última instância (mais recente)
        return self.classifier(h[-1])
```

**Quando usar**: Dados com drift gradual (vizinhos temporais informam predição).

### 3.3 Estimativa Comparativa: GNN vs MLP vs Transformer

**Cenário**: Elec2, 100 modelos, batch=32

| Arquitetura | Parâmetros | Latência | VRAM | Throughput | Acurácia Estimada |
|-------------|------------|----------|------|------------|-------------------|
| MLP [64,32] | 15k | 2ms | 300MB | 400 inst/s | 81.4% (baseline) |
| Feature GNN | 80k | 3.5ms | 600MB | 230 inst/s | 82-83% (+1-2%) |
| Meta-GNN (ensemble) | 50k | 2.2ms | 350MB | 360 inst/s | 83-85% (+2-4%) |
| Temporal GNN (window=32) | 100k | 5ms | 800MB | 160 inst/s | 82-84% (+1-3%) |
| Transformer (window=32) | 250k | 15ms | 5GB | 50 inst/s | 81-82% (+0-1%) |

**Conclusão**: **Meta-GNN oferece melhor custo-benefício** (pequeno overhead, potencial ganho substancial).

---

## 4. PLANO DE IMPLEMENTAÇÃO RECOMENDADO

### Fase 1: GNN como Base Learner (ALTA PRIORIDADE)

**Objetivo**: Substituir 50% dos MLPs por Feature GNNs no ensemble híbrido

```python
# Ensemble híbrido: 50 MLPs + 50 GNNs
ensemble = []

for i in range(100):
    if i < 50:
        # MLP tradicional
        model = FlexibleNeuralNetwork(n_feat, n_classes, [64, 32])
    else:
        # GNN para interações de features
        model = FeatureGraphNN(n_feat, n_classes, hidden_dim=64)
    
    classifier = classification.Classifier(
        module=model,
        optimizer_fn=optim.Adam,
        lr=0.005,
        device='cuda'
    )
    ensemble.append(classifier)
```

**Experimentos Planejados**:
1. Baseline: 100 MLPs
2. Híbrido: 50 MLPs + 50 GNNs
3. Puro: 100 GNNs

**Métricas**:
- Acurácia, Kappa, G-Mean (desempenho)
- Latência, throughput (eficiência)
- Número de drifts detectados (adaptabilidade)

**Prazo**: 2-3 semanas

### Fase 2: Meta-GNN para Agregação (MUITO PROMISSOR)

**Objetivo**: Substituir votação majoritária por Meta-GNN

**Implementação**:
```python
class ARTELightMetaGNN(ARTELight):
    def __init__(self, models, model_types, drift_detector, **kwargs):
        super().__init__(models, model_types, drift_detector, **kwargs)
        
        # Adiciona Meta-GNN
        self.meta_gnn = EnsembleGNN(
            n_models=len(models),
            n_classes=2,
            hidden_dim=64
        ).to('cuda')
        
        self.meta_optimizer = optim.Adam(self.meta_gnn.parameters(), lr=0.001)
    
    def predict_proba_one(self, x):
        # Coleta predições + features dos modelos
        model_states = self._gather_model_states(x)
        
        # Meta-GNN agrega
        final_proba = self.meta_gnn(
            model_states['features'],
            model_states['predictions'],
            self.edge_index
        )
        
        return {i: float(p) for i, p in enumerate(final_proba)}
    
    def learn_one(self, x, y):
        # 1. Treina modelos base (como antes)
        any_drift = super().learn_one(x, y)
        
        # 2. Treina Meta-GNN (meta-learning)
        if self.instances_seen % 10 == 0:  # A cada 10 instâncias
            self._train_meta_gnn(x, y)
        
        return any_drift
```

**Prazo**: 1-2 semanas (após Fase 1)

### Fase 3: Temporal GNN (OPCIONAL)

**Condição**: Apenas se análise de autocorrelação nos dados mostrar padrões temporais fortes.

**Teste Prévio**:
```python
from statsmodels.tsa.stattools import acf

# Analisa autocorrelação das labels
acf_values = acf(y_all, nlags=50)

if acf_values[1:10].mean() > 0.3:  # Autocorrelação significativa
    print("✅ Temporal GNN justificado")
else:
    print("❌ Dados IID, GNN temporal não ajudará")
```

**Prazo**: 2 semanas (se necessário)

---

## 5. VIABILIDADE: ANÁLISE DE RECURSOS

### 5.1 Requisitos de Hardware

**Setup Atual** (MLPs):
- GPU: RTX 3080 (10GB VRAM) ou similar
- RAM: 16GB
- Armazenamento: 50GB

**Setup com GNNs**:
- GPU: RTX 3080 (10GB VRAM) - **SUFICIENTE**
- RAM: 16GB - **SUFICIENTE**
- Armazenamento: 50GB - **SUFICIENTE**

**Setup com Transformers** (não recomendado):
- GPU: A100 (40GB VRAM) - **NECESSÁRIO** (upgrade caro)
- RAM: 32GB - **UPGRADE NECESSÁRIO**
- Armazenamento: 100GB

### 5.2 Custo Computacional Estimado

**Treinamento de 1 Dataset** (100k instâncias):

| Arquitetura | Tempo (GPU) | Custo Energia | VRAM Pico |
|-------------|-------------|---------------|-----------|
| MLP (baseline) | 5 min | ~$0.02 | 300MB |
| GNN (Feature) | 8 min | ~$0.03 | 600MB |
| GNN (Meta) | 5.5 min | ~$0.022 | 350MB |
| Transformer | 45 min | ~$0.20 | 5GB |

**Bateria Completa** (11 datasets, 5 seeds = 55 runs):

| Arquitetura | Tempo Total | Custo Total |
|-------------|-------------|-------------|
| MLP | 4.6 horas | ~$1.10 |
| GNN Feature | 7.3 horas | ~$1.65 |
| GNN Meta | 5 horas | ~$1.21 |
| Transformer | 41 horas | ~$11 |

**Conclusão**: GNNs são viáveis (overhead de 50-60% em tempo), Transformers não.

---

## 6. ROADMAP DE IMPLEMENTAÇÃO (12 SEMANAS)

### Semanas 1-3: Preparação
- [ ] Instalar PyTorch Geometric
- [ ] Implementar FeatureGraphNN básico
- [ ] Testes unitários (forward/backward)
- [ ] Integração com River/Deep-River

### Semanas 4-6: Experimentos GNN como Base Learner
- [ ] Baseline: 100 MLPs (reproduzir resultados)
- [ ] Experimento 1: 100 GNNs
- [ ] Experimento 2: 50 MLPs + 50 GNNs
- [ ] Análise de resultados (acurácia, latência, drift)

### Semanas 7-9: Meta-GNN
- [ ] Implementar EnsembleGNN
- [ ] Integrar com ARTELight
- [ ] Experimentos em 3 datasets
- [ ] Ablation study (diferentes estratégias de grafo)

### Semanas 10-11: Bateria Completa
- [ ] Rodar todos os datasets
- [ ] Comparação estatística (Friedman test)
- [ ] Gráficos e visualizações

### Semana 12: Análise e Documentação
- [ ] Relatório técnico
- [ ] Preparação de artigo
- [ ] Código limpo + README

---

## 7. RESPOSTA DIRETA ÀS SUAS PERGUNTAS

### "Qual o potencial de contribuição?"

**GNNs**: ⭐⭐⭐⭐⭐ (5/5)
- Potencial ganho de +2-4% em acurácia
- Overhead computacional aceitável (+50% tempo)
- Adequados para tabular (relações feature-feature)
- **Meta-GNN especialmente promissor** (melhora agregação do ensemble)

**Transformers**: ⭐⭐ (2/5)
- Ganho esperado: +0-1% (marginal)
- Overhead computacional proibitivo (16x slower)
- Não adequados para tabular IID
- Úteis apenas com autocorrelação temporal forte

### "Seriam competitivas do ponto de vista de latência?"

**GNN Feature**: ⚠️ Moderadamente competitivo
- 3.5ms vs 2ms (MLP) = 75% slower
- Throughput: 230 inst/s vs 400 inst/s
- **Viável para produção** se ganho em acurácia compensar

**GNN Meta**: ✅ Altamente competitivo
- 2.2ms vs 2ms (MLP) = 10% slower
- Throughput: 360 inst/s vs 400 inst/s
- **Excelente custo-benefício**

**Transformer**: ❌ Não competitivo
- 15ms vs 2ms = 650% slower
- Throughput: 50 inst/s vs 400 inst/s
- **Inviável para streaming de alta frequência**

### "Como você implementaria?"

**Passo-a-Passo** (Meta-GNN):

1. **Instalar dependências**:
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

2. **Implementar classes base** (ver código completo acima)

3. **Integrar com ARTELight existente**:
```python
# Modificar apenas o método predict_proba_one
# Resto do código permanece idêntico
```

4. **Treinar progressivamente**:
```python
# Meta-GNN aprende online junto com os base learners
# A cada K instâncias, atualiza pesos do GNN
```

5. **Experimentar em 1 dataset primeiro** (Elec2)

6. **Se bem-sucedido, expandir para bateria completa**

---

## 8. RECOMENDAÇÃO FINAL

### IMPLEMENTAR (ALTA PRIORIDADE):
1. ✅ **Meta-GNN para agregação** - Melhor ROI (retorno sobre investimento)
2. ✅ **Feature GNN como learner alternativo** - Testar diversidade arquitetural

### CONSIDERAR (PRIORIDADE MÉDIA):
3. ⚠️ **Temporal GNN** - Apenas se autocorrelação for detectada

### EVITAR (BAIXA PRIORIDADE):
4. ❌ **Transformers como base learner** - Overhead proibitivo para ganho marginal
5. ⚠️ **Transformer como meta-learner** - Considerar apenas se Meta-GNN falhar

### Justificativa:
Meta-GNN oferece o "melhor dos dois mundos":
- Overhead mínimo (~10% latência)
- Potencial ganho substancial (~2-4% acurácia)
- Interpretabilidade (visualizar grafo do ensemble)
- Inovação científica (poucos trabalhos usam GNN para ensemble streaming)

**Publicação**: Artigo com título "Graph Neural Networks for Adaptive Ensemble Aggregation in Data Streams" teria alto impacto.

---

**Conclusão**: GNNs (especialmente Meta-GNN) são **altamente promissores** e **computacionalmente viáveis**. Transformers **não são recomendados** para este projeto (overhead vs benefício desfavorável).
