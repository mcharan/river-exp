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
# ARTESubspaceNN — ARTE Neural com Random Subspace por membro
# =============================================================================
class ARTESubspaceNN:
    """
    Port neural direto do ARTE usando subespaços aleatórios de features.

    Cada membro do ensemble (MLP):
      - Opera sobre um subconjunto aleatório de k features  (k ∈ [k_min, n_features])
      - Possui seu próprio ADWINChangeDetector individual
      - É reiniciado individualmente (novo MLP + novo subconjunto) ao detectar drift
      - Voto ponderado pela acurácia na janela deslizante (igual ao ARTELight)
      - Treinamento Online Bagging via Poisson(lambd) repetições por instância

    Análogo neural do ARTE original: diversidade via subconjunto de features +
    reset total do membro afetado (não do ensemble inteiro).
    """

    def __init__(self, n_features, n_classes, n_models=30, lambd=6.0,
                 k_min=2, window_size=500, seed=42, hidden_layers=None,
                 lr=0.005, delta=0.001, device='cpu'):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_models = n_models
        self.lambd = lambd
        self.k_min = max(1, min(k_min, n_features))
        self.window_size = window_size
        self.seed = seed
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64]
        self.lr = lr
        self.delta = delta
        self.device = device

        self._rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        self._loss_fn = nn.CrossEntropyLoss()
        self._total_drifts = 0

        self._members = [self._new_member() for _ in range(n_models)]

    def _new_member(self):
        k = self._rng.randint(self.k_min, self.n_features + 1)
        indices = np.sort(self._rng.choice(self.n_features, k, replace=False))
        subspace = torch.tensor(indices, dtype=torch.long, device=self.device)

        model = FlexibleNeuralNetwork(
            n_features=k,
            n_classes=self.n_classes,
            hidden_layers=self.hidden_layers,
        ).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        detector = ADWINChangeDetector(delta=self.delta)
        window_acc = utils.Rolling(stats.Mean(), window_size=self.window_size)

        return {
            'model': model,
            'optimizer': optimizer,
            'subspace': subspace,
            'detector': detector,
            'window_acc': window_acc,
        }

    @torch.inference_mode()
    def predict_proba_one(self, x_full):
        """
        x_full: tensor 1-D de comprimento n_features.
        Retorna Counter {classe: probabilidade_agregada}.
        """
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
            x_sub = x_full[member['subspace']].unsqueeze(0)
            logits = member['model'](x_sub)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
            for cls_idx, prob_val in enumerate(proba):
                votes[cls_idx] += prob_val / n_eligible

        return votes

    def predict_one(self, x_full):
        y_proba = self.predict_proba_one(x_full)
        if y_proba:
            return max(y_proba, key=y_proba.get)
        return 0

    def learn_one(self, x_full, y):
        """
        x_full: tensor 1-D de comprimento n_features.
        y: rótulo inteiro da classe.
        """
        for i, member in enumerate(self._members):
            x_sub = x_full[member['subspace']]
            x_in = x_sub.unsqueeze(0)

            # 1. Predição para monitoramento (test-then-train)
            member['model'].eval()
            with torch.inference_mode():
                logits = member['model'](x_in)
                y_pred = torch.argmax(logits, dim=1).item()

            correct = (y == y_pred)
            member['detector'].update(0 if correct else 1)
            member['window_acc'].update(1 if correct else 0)

            # 2. Poisson(lambd) passos de treinamento
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

            # 3. Drift: reinicia somente este membro (novo subconjunto + novo MLP)
            if member['detector'].drift_detected:
                self._total_drifts += 1
                self._members[i] = self._new_member()

    @property
    def total_drifts(self):
        return self._total_drifts


# =============================================================================
# ARTESoftResetNN — ARTE Neural com Reset Parcial de Camadas por membro
# =============================================================================
class ARTESoftResetNN:
    """
    Igual ao ARTELight (ensemble de MLPs com composição diversificada),
    mas ao detectar drift em um membro reinicia apenas as últimas
    `n_reset_layers` camadas lineares do MLP afetado, preservando as
    camadas de extração de features anteriores.

    Motivação: redes neurais sofrem de esquecimento catastrófico após um
    reset completo — precisam de muitas instâncias para re-convergir.
    O reset suave preserva o conhecimento acumulado nas camadas iniciais
    e reinicia apenas a cabeça de decisão, que é mais sensível ao drift.

    Parâmetros
    ----------
    models : list de deep_river classification.Classifier
    model_configs : list de dict com 'optimizer_fn' e 'lr'
        Usado para recriar o optimizer após o reset parcial.
    drift_detector : ADWINChangeDetector ou NoDriftDetector
    n_reset_layers : int
        Número de camadas lineares finais a reiniciar no drift.
    """

    def __init__(self, models, model_configs, drift_detector,
                 n_reset_layers=1, lambda_val=6, seed=42, window_size=500):
        self.models = models
        self.model_configs = model_configs
        self.drift_detector = drift_detector
        self.n_reset_layers = n_reset_layers
        self.lambda_val = lambda_val
        self.window_size = window_size

        self._rng = np.random.RandomState(seed)
        self._loss_fn = nn.CrossEntropyLoss()
        self._total_drifts = 0

        n = len(models)
        self._detectors = [drift_detector.clone() for _ in range(n)]
        self._acc_windows = [
            utils.Rolling(stats.Mean(), window_size=window_size) for _ in range(n)
        ]

    def _soft_reset(self, i):
        """
        Reinicializa as últimas n_reset_layers camadas lineares do membro i.
        Recria o optimizer para que as referências aos parâmetros sejam válidas.
        """
        model = self.models[i]
        cfg = self.model_configs[i]

        # Coleta todas as camadas nn.Linear dentro de mlp_head
        linear_layers = [
            m for m in model.module.mlp_head.modules()
            if isinstance(m, nn.Linear)
        ]

        # Reinicializa as últimas n_reset_layers
        for layer in linear_layers[-self.n_reset_layers:]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        # Recria optimizer para limpar momentum e referências antigas
        model.optimizer = cfg['optimizer_fn'](
            model.module.parameters(), lr=cfg['lr']
        )

    @torch.inference_mode()
    def predict_proba_one(self, x):
        """
        x: tensor [n_feat] ou [1, n_feat].
        Retorna Counter {classe: probabilidade_agregada}.
        """
        if isinstance(x, torch.Tensor) and x.ndim == 1:
            x = x.unsqueeze(0)

        accs = [w.get() for w in self._acc_windows]
        avg = sum(accs) / len(accs) if accs else 0.0
        eligible = [i for i, a in enumerate(accs) if a >= avg]
        if not eligible:
            eligible = list(range(len(self.models)))

        votes = collections.Counter()
        n_eligible = len(eligible)

        for i in eligible:
            model = self.models[i]
            model.module.eval()
            logits = model.module(x)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
            for cls_idx, prob_val in enumerate(proba):
                votes[cls_idx] += prob_val / n_eligible

        return votes

    def predict_one(self, x):
        y_proba = self.predict_proba_one(x)
        if y_proba:
            return max(y_proba, key=y_proba.get)
        return 0

    def learn_one(self, x, y):
        """
        x: tensor [n_feat] ou [1, n_feat].
        y: rótulo inteiro da classe.
        """
        if isinstance(x, torch.Tensor) and x.ndim == 1:
            x_in = x.unsqueeze(0)
        else:
            x_in = x

        for i, model in enumerate(self.models):
            # 1. Predição para monitoramento (test-then-train)
            model.module.eval()
            with torch.inference_mode():
                logits = model.module(x_in)
                y_pred = torch.argmax(logits, dim=1).item()

            correct = (y == y_pred)
            self._detectors[i].update(0 if correct else 1)
            self._acc_windows[i].update(1 if correct else 0)

            # 2. Poisson boosting nas predições incorretas
            if not correct:
                k = self._rng.poisson(self.lambda_val)
                if k > 0:
                    model.module.train()
                    x_boost = x_in.repeat(k, 1)
                    y_boost = torch.tensor(
                        [y] * k, device=x_in.device, dtype=torch.long
                    )
                    model.optimizer.zero_grad()
                    logits_train = model.module(x_boost)
                    loss = self._loss_fn(logits_train, y_boost)
                    loss.backward()
                    model.optimizer.step()

            # 3. Drift: reset parcial somente deste membro
            if self._detectors[i].drift_detected:
                self._total_drifts += 1
                self._soft_reset(i)
                self._detectors[i] = self.drift_detector.clone()
                self._acc_windows[i] = utils.Rolling(
                    stats.Mean(), window_size=self.window_size
                )

    @property
    def total_drifts(self):
        return self._total_drifts
