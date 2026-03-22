"""
Ensemble-GNN para NeuralARTE
============================
Estratégia 2 da análise (20260208-ANALISE_TRANSFORMERS_GNNS.md):
Cada modelo do ensemble é um nó no grafo. A GNN aprende a agregar
as predições dos modelos de forma adaptativa, substituindo a votação
majoritária por uma agregação guiada pelo grafo.

Arquitetura:
  - Nós: cada modelo do ensemble (n_models nós)
  - Arestas: vizinhança completa (todos os pares) — ou k-NN por similaridade
  - Features dos nós: [predição_softmax | last_loss | drift_flag] (dim = n_classes+2)
  - Camadas: GATConv (atenção → o grafo aprende quais modelos "ouvir")
  - Saída: logits de classe para o ensemble

Por que GATConv?
  Atenção por aresta permite ao grafo ignorar modelos que driftaram recentemente
  sem precisar de um mecanismo externo de pesagem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Importação opcional do PyTorch Geometric
try:
    from torch_geometric.nn import GATConv, GCNConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[GNN] PyTorch Geometric nao disponivel. Use EnsembleGNNFallback.")


# =============================================================================
# UTILITÁRIO: construção do grafo do ensemble
# =============================================================================

def build_full_graph_edges(n_models: int, device: torch.device) -> torch.Tensor:
    """
    Retorna edge_index (2 x E) para grafo completo com self-loops.
    E = n_models^2 (inclui auto-conexão de cada nó).
    """
    src, dst = [], []
    for i in range(n_models):
        for j in range(n_models):
            src.append(i)
            dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index


def build_knn_graph_edges(similarities: torch.Tensor, k: int, device: torch.device) -> torch.Tensor:
    """
    Constrói grafo k-NN a partir de matriz de similaridade entre modelos.
    similarities: (n_models, n_models) — ex: produto interno das predições
    """
    n = similarities.size(0)
    # Top-k vizinhos por linha (excluindo o próprio nó)
    topk = torch.topk(similarities, k=min(k + 1, n), dim=1).indices
    src, dst = [], []
    for i in range(n):
        for j in topk[i]:
            if j != i:
                src.append(i)
                dst.append(j.item())
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index


# =============================================================================
# MODELO: EnsembleGNN com GATConv
# =============================================================================

class EnsembleGNN(nn.Module):
    """
    Meta-GNN que agrega predições do ensemble via Graph Attention Network.

    Parâmetros
    ----------
    n_models : int
        Número de modelos base no ensemble.
    n_classes : int
        Número de classes do problema.
    hidden_dim : int
        Dimensão oculta das camadas GATConv.
    n_heads : int
        Número de cabeças de atenção.
    dropout : float
        Dropout nas camadas de atenção.

    Features de cada nó (modelo):
        [ softmax[0..n_classes-1] | last_loss | drift_flag ]
        dim total = n_classes + 2
    """

    def __init__(self, n_models: int, n_classes: int, hidden_dim: int = 64,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert HAS_PYG, "PyTorch Geometric é necessário para EnsembleGNN"

        self.n_models = n_models
        self.n_classes = n_classes
        self.node_feat_dim = n_classes + 2   # softmax + last_loss + drift_flag

        # Projeção de entrada
        self.input_proj = nn.Linear(self.node_feat_dim, hidden_dim)

        # Camadas GAT
        self.gat1 = GATConv(hidden_dim, hidden_dim // n_heads,
                            heads=n_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // n_heads,
                            heads=n_heads, dropout=dropout, concat=True)

        # Cabeça de classificação global
        # Usa readout por mean pooling sobre todos os nós → logits de classe
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

        self.dropout = dropout

    def forward(self, node_features: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros
        ----------
        node_features : (n_models, node_feat_dim)
        edge_index    : (2, E)

        Retorna
        -------
        logits : (1, n_classes)  — para classificação da instância atual
        """
        x = self.input_proj(node_features)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Mean pooling sobre todos os nós
        x_global = x.mean(dim=0, keepdim=True)   # (1, hidden_dim)
        logits = self.head(x_global)               # (1, n_classes)
        return logits


# =============================================================================
# FALLBACK: agregação simples quando PYG não está disponível
# =============================================================================

class EnsembleGNNFallback(nn.Module):
    """
    Versão sem PyTorch Geometric: simula a meta-agregação com uma MLP
    que recebe a concatenação das predições de todos os modelos.

    Útil para verificar o pipeline sem instalar torch_geometric.
    node_features: (n_models, n_classes+2) → flatten → MLP → (n_classes,)
    """

    def __init__(self, n_models: int, n_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.n_models = n_models
        self.n_classes = n_classes
        self.node_feat_dim = n_classes + 2
        in_dim = n_models * self.node_feat_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, node_features: torch.Tensor,
                edge_index=None) -> torch.Tensor:
        """
        node_features: (n_models, node_feat_dim)
        Retorna logits: (1, n_classes)
        """
        x = node_features.flatten().unsqueeze(0)   # (1, n_models * feat_dim)
        return self.net(x)


# =============================================================================
# WRAPPER: gerenciamento online do Meta-GNN
# =============================================================================

class MetaGNNAggregator:
    """
    Gerencia o grafo do ensemble, coleta features dos nós e treina o Meta-GNN
    de forma online a cada K instâncias.

    Uso típico no loop principal do NeuralARTE:
        aggregator = MetaGNNAggregator(n_models, n_classes, device)

        # para cada instância x_t:
        node_feats = aggregator.collect_node_features(probas_list, losses, drifts)
        pred_class  = aggregator.predict(node_feats)
        aggregator.update(node_feats, true_label)   # treina a cada K instâncias
    """

    def __init__(self, n_models: int, n_classes: int, device: torch.device,
                 hidden_dim: int = 64, n_heads: int = 4, lr: float = 1e-3,
                 update_every: int = 10, graph_type: str = "full"):
        self.n_models = n_models
        self.n_classes = n_classes
        self.device = device
        self.update_every = update_every
        self.instances_seen = 0

        # Escolhe implementação conforme disponibilidade
        if HAS_PYG:
            self.model = EnsembleGNN(n_models, n_classes, hidden_dim, n_heads).to(device)
        else:
            print("[MetaGNN] Usando fallback MLP (instale torch_geometric para GATConv)")
            self.model = EnsembleGNNFallback(n_models, n_classes, hidden_dim).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        # Grafo pré-calculado (full) ou recalculado (knn)
        self.graph_type = graph_type
        if graph_type == "full":
            self.edge_index = build_full_graph_edges(n_models, device)
        else:
            self.edge_index = None   # calculado dinamicamente em knn mode

        # Buffer de treinamento (guarda últimas K instâncias)
        self._buf_feats: list = []
        self._buf_labels: list = []

    # ------------------------------------------------------------------

    def collect_node_features(self,
                               probas_list: list,
                               losses: list,
                               drifts: list) -> torch.Tensor:
        """
        Constrói o tensor de features dos nós a partir do estado atual dos modelos.

        Parâmetros
        ----------
        probas_list : list[list[float]]  — softmax de cada modelo, len=n_models
        losses      : list[float]        — last loss de cada modelo
        drifts      : list[int]          — 1 se driftou recentemente, 0 c.c.

        Retorna
        -------
        node_features : (n_models, n_classes+2) tensor no device
        """
        rows = []
        for i in range(self.n_models):
            proba = list(probas_list[i])[:self.n_classes]
            # padding por segurança
            while len(proba) < self.n_classes:
                proba.append(0.0)
            row = proba + [float(losses[i]), float(drifts[i])]
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------

    def predict(self, node_features: torch.Tensor) -> int:
        """Retorna a classe predita (int) para a instância atual."""
        self.model.eval()
        with torch.no_grad():
            if self.graph_type == "knn":
                sims = torch.mm(node_features[:, :self.n_classes],
                                node_features[:, :self.n_classes].T)
                self.edge_index = build_knn_graph_edges(sims, k=5, device=self.device)
            logits = self.model(node_features, self.edge_index)
            return int(logits.argmax(dim=1).item())

    # ------------------------------------------------------------------

    def update(self, node_features: torch.Tensor, true_label: int):
        """
        Acumula instância no buffer e treina o Meta-GNN a cada `update_every` inst.
        """
        self.instances_seen += 1
        self._buf_feats.append(node_features.detach().cpu())
        self._buf_labels.append(true_label)

        # Treina quando o buffer atinge o tamanho alvo
        if len(self._buf_labels) >= self.update_every:
            self._train_step()
            self._buf_feats.clear()
            self._buf_labels.clear()

    # ------------------------------------------------------------------

    def _train_step(self):
        """Realiza uma passagem de gradiente sobre o buffer acumulado."""
        self.model.train()
        total_loss = 0.0
        for feats, label in zip(self._buf_feats, self._buf_labels):
            feats = feats.to(self.device)
            logits = self.model(feats, self.edge_index)
            target = torch.tensor([label], dtype=torch.long, device=self.device)
            loss = self.criterion(logits, target)
            total_loss += loss

        self.optimizer.zero_grad()
        (total_loss / len(self._buf_labels)).backward()
        self.optimizer.step()
