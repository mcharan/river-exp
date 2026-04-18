import torch
from torch import nn, optim
import numpy as np
import collections
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.arte.drift_detector import ADWINChangeDetector
from src.neural_arte.neural_arte import FlexibleNeuralNetwork
from river import stats, utils


# =============================================================================
# Composicoes de ensemble — alinhadas com NeuralARTE para comparacao direta
# Cada tier eh um dict de configuracao de membro.
# Para n_models membros, os tiers sao ciclados (round-robin).
# =============================================================================
def _make_rotation_matrix(n_features, device='cpu'):
    q, _ = torch.linalg.qr(torch.randn(n_features, n_features))
    return q.to(device)


COMPOSITIONS = {
    # Tiers abc: Fast(SGD) + Deep(Adam) + Mid(Adam), sem CNN nem projecao
    'abc': [
        {'use_cnn': False, 'use_projection': False, 'optimizer': 'sgd',  'lr': 0.01,  'hidden_layers': [64]},
        {'use_cnn': False, 'use_projection': False, 'optimizer': 'adam', 'lr': 0.001, 'hidden_layers': [128, 64, 32]},
        {'use_cnn': False, 'use_projection': False, 'optimizer': 'adam', 'lr': 0.001, 'hidden_layers': [64, 32]},
    ],
    # abc com projecao ortogonal por membro (diversidade de subespaco)
    'abc_proj': [
        {'use_cnn': False, 'use_projection': True, 'optimizer': 'sgd',  'lr': 0.01,  'hidden_layers': [64]},
        {'use_cnn': False, 'use_projection': True, 'optimizer': 'adam', 'lr': 0.001, 'hidden_layers': [128, 64, 32]},
        {'use_cnn': False, 'use_projection': True, 'optimizer': 'adam', 'lr': 0.001, 'hidden_layers': [64, 32]},
    ],
    # 4 perfis heterogeneos do notebook original (CNN/no-CNN x SGD/Adam)
    'heterogeneous': [
        {'use_cnn': False, 'use_projection': False, 'optimizer': 'sgd',  'lr': 0.1,   'hidden_layers': [32]},
        {'use_cnn': False, 'use_projection': True,  'optimizer': 'adam', 'lr': 0.001, 'hidden_layers': [64, 32]},
        {'use_cnn': True,  'use_projection': False, 'optimizer': 'sgd',  'lr': 0.05,  'hidden_layers': [32]},
        {'use_cnn': True,  'use_projection': True,  'optimizer': 'adam', 'lr': 0.001, 'hidden_layers': [32]},
    ],
}


# =============================================================================
# HeterogeneousOnlineBagging — Online Bagging heterogeneo com ADWIN direcional
# =============================================================================
class HeterogeneousOnlineBagging:
    """
    Ensemble heterogeneo de MLPs com Online Bagging e deteccao de drift
    por membro usando ADWINChangeDetector direcional (sem falsos positivos
    pos-reset).

    Diferencas vs notebook Bagging-Heterogeneous.ipynb:
    - Usa ADWINChangeDetector (direcional) em vez de river.drift.ADWIN
    - Interface de dataset via get_dataset_universal()
    - Metricas: Accuracy + KappaM
    - Logging via log_results_to_csv()
    - Composicoes alinhadas com NeuralARTE para comparacao direta

    Parametros
    ----------
    n_features : int
    n_classes : int
    composition : str
        Uma das chaves de COMPOSITIONS.
    n_models : int
        Numero de membros do ensemble (tiers ciclados round-robin).
    lambd : float
        Parametro Poisson para online bagging (padrao 6, igual ao NeuralARTE).
    window_size : int
        Janela para acuracia ponderada no voto.
    seed : int
    delta : float
        Parametro ADWIN.
    device : str
        'cuda' ou 'cpu'.
    """

    def __init__(self, n_features, n_classes, composition='abc', n_models=30,
                 lambd=6.0, window_size=500, seed=42, delta=0.001, device='cpu'):
        assert composition in COMPOSITIONS, (
            f"Composicao '{composition}' desconhecida. Disponiveis: {list(COMPOSITIONS.keys())}"
        )
        self.n_features = n_features
        self.n_classes = n_classes
        self.composition = composition
        self.n_models = n_models
        self.lambd = lambd
        self.window_size = window_size
        self.seed = seed
        self.delta = delta
        self.device = device

        self._rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self._loss_fn = nn.CrossEntropyLoss()
        self._total_drifts = 0

        self._members = [self._new_member(i) for i in range(n_models)]

    def _tier_cfg(self, index):
        tiers = COMPOSITIONS[self.composition]
        return tiers[index % len(tiers)]

    def _new_member(self, index):
        cfg = self._tier_cfg(index)

        proj = None
        if cfg['use_projection']:
            proj = _make_rotation_matrix(self.n_features, self.device)

        model = FlexibleNeuralNetwork(
            n_features=self.n_features,
            n_classes=self.n_classes,
            hidden_layers=cfg['hidden_layers'],
            use_cnn=cfg['use_cnn'],
            projection_matrix=proj,
        ).to(self.device)

        opt_cls = optim.SGD if cfg['optimizer'] == 'sgd' else optim.Adam
        optimizer = opt_cls(model.parameters(), lr=cfg['lr'])

        return {
            'model': model,
            'optimizer': optimizer,
            'cfg_index': index,
            'detector': ADWINChangeDetector(delta=self.delta),
            'window_acc': utils.Rolling(stats.Mean(), window_size=self.window_size),
        }

    @torch.inference_mode()
    def predict_proba_one(self, x):
        """
        x: tensor 1-D [n_features] na device correta.
        Retorna Counter {classe: probabilidade_agregada}.
        Voto ponderado pela acuracia (membros acima da media).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        accs = [m['window_acc'].get() for m in self._members]
        avg = sum(accs) / len(accs) if accs else 0.0
        eligible = [i for i, a in enumerate(accs) if a >= avg]
        if not eligible:
            eligible = list(range(len(self._members)))

        votes = collections.Counter()
        n_eligible = len(eligible)
        for i in eligible:
            member = self._members[i]
            member['model'].eval()
            logits = member['model'](x)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
            for cls_idx, prob_val in enumerate(proba):
                votes[cls_idx] += prob_val / n_eligible

        return votes

    def predict_one(self, x):
        y_proba = self.predict_proba_one(x)
        return max(y_proba, key=y_proba.get) if y_proba else 0

    def learn_one(self, x, y):
        """
        x: tensor 1-D [n_features] na device correta.
        y: rotulo inteiro da classe.
        """
        if x.ndim == 1:
            x_in = x.unsqueeze(0)
        else:
            x_in = x

        for i, member in enumerate(self._members):
            # 1. Predicao para monitoramento (test-then-train)
            member['model'].eval()
            with torch.inference_mode():
                logits = member['model'](x_in)
                y_pred = torch.argmax(logits, dim=1).item()

            correct = (y == y_pred)
            member['detector'].update(0 if correct else 1)
            member['window_acc'].update(1 if correct else 0)

            # 2. Online Bagging: Poisson(lambd) repeticoes em TODAS as instancias
            k = self._rng.poisson(self.lambd)
            if k > 0:
                member['model'].train()
                x_boost = x_in.repeat(k, 1)
                y_boost = torch.tensor([y] * k, device=self.device, dtype=torch.long)
                member['optimizer'].zero_grad()
                logits_train = member['model'](x_boost)
                loss = self._loss_fn(logits_train, y_boost)
                loss.backward()
                member['optimizer'].step()

            # 3. Drift: reinicializa membro completo (novo modelo + novo detector)
            if member['detector'].drift_detected:
                self._total_drifts += 1
                self._members[i] = self._new_member(member['cfg_index'])

    @property
    def total_drifts(self):
        return self._total_drifts
