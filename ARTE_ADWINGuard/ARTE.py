import random
import numbers
import collections
import statistics
import numpy as np
from river import base, tree, drift, utils, stats, metrics
from river.tree.splitter import GaussianSplitter
from river.tree.splitter.nominal_splitter_classif import NominalSplitterClassif
from river.tree.nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
from river.tree.utils import BranchFactory

class ARTEGaussianSplitter(GaussianSplitter):
    """
    Implementação fiel ao ARTEAttributeClassObserver.java.
    Em vez de testar vários pontos de corte candidatos, sorteia UM ponto aleatório
    entre o min e max observados e calcula o mérito apenas para ele.
    """
    def __init__(self, rng):
        super().__init__()
        self.rng = rng
        self._min_value = float('inf')
        self._max_value = float('-inf')

    def update(self, att_val, target_val, sample_weight):
        if att_val < self._min_value: self._min_value = att_val
        if att_val > self._max_value: self._max_value = att_val
        super().update(att_val, target_val, sample_weight)

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        if self._min_value >= self._max_value:
            return BranchFactory()

        split_value = self.rng.uniform(self._min_value, self._max_value)
        post_split_dist = self._class_dists_from_binary_split(split_value)
        merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
        return BranchFactory(merit, att_idx, split_value, post_split_dist)


# =============================================================================
# 1. MIXIN DE SUBESPAÇO ALEATÓRIO E INJEÇÃO DE SPLITTER
# =============================================================================
class RandomSubspaceNodeMixin:
    """
    Mixin que implementa:
    1. Seleção de subespaço aleatório (RandomLearningNode.java)
    2. Injeção do Splitter Aleatório (ARTEAttributeClassObserver.java)
    """
    def __init__(self, subspace_size, rng, **kwargs):
        super().__init__(**kwargs)
        self.subspace_size = subspace_size
        self.rng = rng
        self.selected_features = None

    def learn_one(self, x, y, *, w=1.0, tree=None):
        if self.selected_features is None:
            all_features = list(x.keys())
            k = max(1, min(self.subspace_size, len(all_features)))
            self.selected_features = self.rng.sample(all_features, k)

        x_subset = {key: x[key] for key in self.selected_features if key in x}

        nom_attrs = tree.nominal_attributes if (tree and hasattr(tree, 'nominal_attributes')) else []
        for att_id, att_val in x_subset.items():
            if att_id not in self.splitters:
                is_nominal = (
                    not isinstance(att_val, numbers.Number)
                    or isinstance(att_val, bool)
                    or att_id in nom_attrs
                )
                self.splitters[att_id] = (
                    NominalSplitterClassif() if is_nominal
                    else ARTEGaussianSplitter(rng=self.rng)
                )

        super().learn_one(x_subset, y, w=w, tree=tree)

# =============================================================================
# 2. CLASSES DE NÓS CONCRETAS (MC, NB, NBA)
# =============================================================================
class ARTELeafMajorityClass(RandomSubspaceNodeMixin, LeafMajorityClass):
    """No Majority Class com Subespaco Aleatorio."""
    pass

class ARTELeafNaiveBayes(RandomSubspaceNodeMixin, LeafNaiveBayes):
    """No Naive Bayes com Subespaco Aleatorio."""
    pass

class ARTELeafNaiveBayesAdaptive(RandomSubspaceNodeMixin, LeafNaiveBayesAdaptive):
    """No Naive Bayes Adaptive com Subespaco Aleatorio."""
    pass

class ARTEHoeffdingTree(tree.HoeffdingTreeClassifier):
    """
    Port do ARTEHoeffdingTree.java para River.
    """
    def __init__(self, subspace_size=2, seed=None, **kwargs):
        kwargs['remove_poor_attrs'] = False
        super().__init__(**kwargs)
        self.subspace_size = subspace_size
        self._rng = random.Random(seed)

    def _new_leaf(self, initial_stats=None, parent=None):
        """Sobrescreve _new_leaf do River para criar nós com RandomSubspaceNodeMixin."""
        if initial_stats is None:
            initial_stats = {}
        depth = 0 if parent is None else parent.depth + 1

        if self.leaf_prediction == 'mc':
            node_cls = ARTELeafMajorityClass
        elif self.leaf_prediction == 'nb':
            node_cls = ARTELeafNaiveBayes
        else:  # 'nba' (default)
            node_cls = ARTELeafNaiveBayesAdaptive

        return node_cls(
            subspace_size=self.subspace_size,
            rng=self._rng,
            stats=initial_stats,
            depth=depth,
            splitter=self.splitter,
        )

class ARTE(base.Ensemble, base.Classifier):
    """Adaptive Random Tree Ensemble (ARTE) portado do MOA — variante com ADWIN Guard.

    Adiciona o parâmetro `drift_detection_grace_period`: número mínimo de instâncias
    que uma árvore deve processar após um reset antes de o detector de drift ser
    atualizado novamente.

    Motivação: o river.drift.ADWIN possui min_window_length=5 (vs 10 no MOA), o que
    causa falsos positivos imediatamente após resets em datasets com drift contínuo
    (ex: rbf_m, rbf_f, agrawal). O guard evita a espiral de resets sincronizados
    com o clock do ADWIN (32 instâncias), sem modificar a lógica do algoritmo base.

    Parâmetros adicionais vs ARTE original:
        drift_detection_grace_period (int): instâncias de carência após reset antes
            de reativar o detector. Default=200 (aprox. 6 clock ticks do ADWIN).
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
        k_min: int = 2,
        drift_detection_grace_period: int = 200,
    ):

        self.n_features = n_features
        self.nominal_attributes = nominal_attributes or []
        self.n_models = n_models
        self.lambd = lambd
        self.drift_detector = drift_detector or drift.ADWIN(delta=1e-3)
        self.window_size = window_size
        self.seed = seed
        self.k_min = k_min
        self.drift_detection_grace_period = drift_detection_grace_period
        self._rng = np.random.RandomState(self.seed)

        self._ensemble_members = []
        for i in range(self.n_models):
            tree_seed = self._rng.randint(0, 1000000)
            k_init = self._rng.randint(self.k_min, self.n_features + 1)

            tree_model = ARTEHoeffdingTree(
                subspace_size=k_init,
                seed=tree_seed,
                nominal_attributes=self.nominal_attributes,
                leaf_prediction='nba',
                grace_period=100,
                delta=0.01
            )

            m = {
                'model': tree_model,
                'detector': self.drift_detector.clone(),
                'window_acc': utils.Rolling(stats.Mean(), window_size=self.window_size),
                'instances_trained': 0,
                'instances_since_reset': drift_detection_grace_period,  # começa já "aquecido"
            }
            self._ensemble_members.append(m)

        super().__init__(models=[m['model'] for m in self._ensemble_members])
        self._avg_window_acc = 0.0
        self._total_drifts = 0

    def learn_one(self, x, y):
        all_accs = []

        for m in self._ensemble_members:
            k = self._rng.poisson(self.lambd)
            if k > 0:
                m['model'].learn_one(x, y, w=k)
                m['instances_trained'] += 1

            y_pred = m['model'].predict_one(x)
            correct = (y == y_pred)

            # Guard: só atualiza e verifica o detector após o período de carência
            m['instances_since_reset'] += 1
            if m['instances_since_reset'] >= self.drift_detection_grace_period:
                m['detector'].update(0 if correct else 1)
                if m['detector'].drift_detected:
                    self._total_drifts += 1
                    self._reset_member(m)

            m['window_acc'].update(1 if correct else 0)
            all_accs.append(m['window_acc'].get())

        if all_accs:
            self._avg_window_acc = statistics.mean(all_accs)

        return self

    def predict_proba_one(self, x):
        combined_votes = collections.Counter()

        eligible_members = [
            m for m in self._ensemble_members
            if self.window_size == 0 or m['window_acc'].get() >= self._avg_window_acc
        ]

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
        return 0

    def _reset_member(self, m):
        """Reinicia o modelo, detector e contador de carência."""
        new_k = self._rng.randint(self.k_min, self.n_features + 1)
        new_seed = self._rng.randint(0, 1000000)

        m['model'] = ARTEHoeffdingTree(
            subspace_size=new_k,
            seed=new_seed,
            nominal_attributes=self.nominal_attributes,
            leaf_prediction='nba',
            grace_period=100,
            delta=0.01
        )
        m['detector'] = self.drift_detector.clone()
        m['window_acc'] = utils.Rolling(stats.Mean(), window_size=self.window_size)
        m['instances_since_reset'] = 0  # reinicia o contador de carência

    @property
    def total_drifts(self):
        return self._total_drifts
