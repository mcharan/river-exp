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
# ARTENeuralRS — Random Subspace Neural Ensemble
# =============================================================================
class ARTENeuralRS:
    """
    Direct neural port of ARTE using random subspaces.

    Each MLP member:
      - Uses a random subspace of k features (k drawn from [k_min, n_features])
      - Has its own ADWINChangeDetector
      - Gets individually reset (new MLP + new random subspace) when drift detected
      - Weighted voting by rolling window accuracy (same as ARTELight)
      - Online Bagging via Poisson(lambd) training steps
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
        x_full: 1-D GPU tensor of length n_features.
        Returns Counter {class_idx: aggregated_prob}.
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
        x_full: 1-D GPU tensor of length n_features.
        y: integer class label.
        """
        for i, member in enumerate(self._members):
            x_sub = x_full[member['subspace']]
            x_in = x_sub.unsqueeze(0)

            # 1. Predict for monitoring (test-then-train)
            member['model'].eval()
            with torch.inference_mode():
                logits = member['model'](x_in)
                y_pred = torch.argmax(logits, dim=1).item()

            correct = (y == y_pred)
            member['detector'].update(0 if correct else 1)
            member['window_acc'].update(1 if correct else 0)

            # 2. Poisson(lambd) training steps
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

            # 3. Drift detection — reset member on drift
            if member['detector'].drift_detected:
                self._total_drifts += 1
                self._members[i] = self._new_member()

    @property
    def total_drifts(self):
        return self._total_drifts


# =============================================================================
# ARTENeuralSR — Selective Reset Neural Ensemble
# =============================================================================
class ARTENeuralSR:
    """
    Same structure as ARTELight but instead of cloning the whole model on drift,
    resets only the last `n_reset_layers` linear layers of the affected model.
    This preserves early feature extraction layers while re-initializing the
    decision layers.

    models: list of deep_river classification.Classifier instances.
    model_configs: list of dicts with keys 'optimizer_fn' and 'lr',
                   used to recreate the optimizer after selective reset.
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

    def _selective_reset(self, i):
        """
        Reinitialise the last n_reset_layers Linear layers of model i's mlp_head.
        Then recreate the optimizer so parameter references stay valid.
        """
        model = self.models[i]
        cfg = self.model_configs[i]

        # Collect all nn.Linear layers inside mlp_head
        linear_layers = [
            m for m in model.module.mlp_head.modules()
            if isinstance(m, nn.Linear)
        ]

        # Reinit the last n_reset_layers linear layers
        for layer in linear_layers[-self.n_reset_layers:]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        # Recreate optimizer so it holds fresh references to the (modified) params
        model.optimizer = cfg['optimizer_fn'](
            model.module.parameters(), lr=cfg['lr']
        )

    @torch.inference_mode()
    def predict_proba_one(self, x):
        """
        x: GPU tensor [n_feat] or [1, n_feat].
        Returns Counter {class_idx: aggregated_prob}.
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
        x: GPU tensor [n_feat] or [1, n_feat].
        y: integer class label.
        """
        if isinstance(x, torch.Tensor) and x.ndim == 1:
            x_in = x.unsqueeze(0)
        else:
            x_in = x

        for i, model in enumerate(self.models):
            # 1. Predict for monitoring (test-then-train)
            model.module.eval()
            with torch.inference_mode():
                logits = model.module(x_in)
                y_pred = torch.argmax(logits, dim=1).item()

            correct = (y == y_pred)
            self._detectors[i].update(0 if correct else 1)
            self._acc_windows[i].update(1 if correct else 0)

            # 2. Poisson boosting on incorrect predictions
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

            # 3. Selective reset on drift
            if self._detectors[i].drift_detected:
                self._total_drifts += 1
                self._selective_reset(i)
                self._detectors[i] = self.drift_detector.clone()
                self._acc_windows[i] = utils.Rolling(
                    stats.Mean(), window_size=self.window_size
                )

    @property
    def total_drifts(self):
        return self._total_drifts
