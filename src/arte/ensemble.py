import random
import numbers
import collections
import statistics
import numpy as np
from river import base, utils, stats

from .drift_detector import ADWINChangeDetector
from .tree import ARTEHoeffdingTree


class ARTE(base.Ensemble, base.Classifier):
    """Adaptive Random Tree Ensemble (ARTE) portado do MOA.

    Algoritmo adaptativo para fluxos de dados evolutivos de Paim e Enembreck.
    """

    def __init__(
        self,
        n_features: int,
        nominal_attributes: list = None,
        n_models: int = 100,
        lambd: float = 6.0,
        drift_detector: base.DriftDetector = None,
        window_size: int = 1000,
        seed: int = 1,
        k_min: int = 2
    ):

        self.n_features = n_features
        self.nominal_attributes = nominal_attributes or [] # Lista de índices
        self.n_models = n_models
        self.lambd = lambd
        self.drift_detector = drift_detector or ADWINChangeDetector(delta=1e-3)
        self.window_size = window_size
        self.seed = seed
        self.k_min = k_min
        self._rng = np.random.RandomState(self.seed)

        # Inicialização dos membros conforme a estrutura AREBaseLearner do original
        self._ensemble_members = []
        for i in range(self.n_models):
            tree_seed = self._rng.randint(0, 1000000)

            # Sorteia k inicial
            k_init = self._rng.randint(self.k_min, self.n_features + 1)

            # Cria árvore com Random Subspace e SEM Poda
            tree_model = ARTEHoeffdingTree(
                subspace_size=k_init,
                seed=tree_seed,
                nominal_attributes=self.nominal_attributes,
                leaf_prediction='nba',  # NBAdaptive — default do MOA HoeffdingTree (índice 2)
                grace_period=100,
                delta=0.01
            )

            m = {
                'model': tree_model,
                'detector': self.drift_detector.clone(),
                'window_acc': utils.Rolling(stats.Mean(), window_size=self.window_size),
                'instances_trained': 0
            }
            self._ensemble_members.append(m)

        super().__init__(models=[m['model'] for m in self._ensemble_members])
        self._avg_window_acc = 0.0
        self._total_drifts = 0

    def learn_one(self, x, y):
        all_accs = []

        for m in self._ensemble_members:
            # Online Bagging via Poisson — sempre sorteado, para todos os membros (fiel ao Java)
            k = self._rng.poisson(self.lambd)
            if k > 0:
                # Uma única chamada com w=k (equivalente ao Java:
                # weightedInstance.setWeight(instance.weight() * k))
                m['model'].learn_one(x, y, w=k)
                m['instances_trained'] += 1

            # Verifica acerto APÓS o treinamento (fiel ao Java: correctlyClassifies
            # é chamado depois de trainOnInstance dentro de ARTEBaseLearner)
            y_pred = m['model'].predict_one(x)
            correct = (y == y_pred)

            # Detecção de Drift individual
            # ADWINChangeDetector replica o comportamento do MOA:
            # só reseta quando o erro aumentou (ver docstring da classe).
            m['detector'].update(0 if correct else 1)
            if m['detector'].drift_detected:
                self._total_drifts += 1
                self._reset_member(m)

            # Atualiza estatísticas da janela deslizante
            m['window_acc'].update(1 if correct else 0)
            all_accs.append(m['window_acc'].get())

        # Atualiza média global para critério de votação seletiva
        if all_accs:
            self._avg_window_acc = statistics.mean(all_accs)

        return self

    def predict_proba_one(self, x):
        combined_votes = collections.Counter()

        # O ARTE filtra votantes cuja acurácia na janela é inferior à média global
        eligible_members = [
            m for m in self._ensemble_members
            if self.window_size == 0 or m['window_acc'].get() >= self._avg_window_acc
        ]

        # Fallback se ninguém estiver acima da média (ex: início do stream)
        if not eligible_members:
            eligible_members = self._ensemble_members

        for m in eligible_members:
            votes = m['model'].predict_proba_one(x)
            if votes:
                total = sum(votes.values())
                if total > 0:
                    for cls, prob in votes.items():
                        combined_votes[cls] += prob / total

        return combined_votes

    def predict_one(self, x):
        proba = self.predict_proba_one(x)
        if proba:
            return max(proba, key=proba.get)
        return 0 # Fallback

    def _reset_member(self, m):
        """
        Reset fiel ao artigo: Sorteia novo k entre [k_min, f].
        """
        # Sorteia novo tamanho de subespaço
        new_k = self._rng.randint(self.k_min, self.n_features + 1)
        new_seed = self._rng.randint(0, 1000000)

        """Reinicia o modelo e estatísticas após detecção de mudança."""
        # Recria a árvore limpa
        m['model'] = ARTEHoeffdingTree(
            subspace_size=new_k,
            seed=new_seed,
            nominal_attributes=self.nominal_attributes,
            leaf_prediction='nba',  # NBAdaptive — default do MOA HoeffdingTree (índice 2)
            grace_period=100,
            delta=0.01
        )
        m['detector'] = self.drift_detector.clone()
        m['window_acc'] = utils.Rolling(stats.Mean(), window_size=self.window_size)

    @property
    def total_drifts(self):
        return self._total_drifts
