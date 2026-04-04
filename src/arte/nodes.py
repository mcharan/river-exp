import numbers
from river.tree.splitter.nominal_splitter_classif import NominalSplitterClassif
from river.tree.nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive

from .splitter import ARTEGaussianSplitter


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
        # 1. Seleção de subespaço aleatório (fixa por nó, como no Java RandomLearningNode)
        if self.selected_features is None:
            all_features = list(x.keys())
            k = max(1, min(self.subspace_size, len(all_features)))
            self.selected_features = self.rng.sample(all_features, k)

        # 2. Filtragem de features
        x_subset = {key: x[key] for key in self.selected_features if key in x}

        # 3. Injeção de splitters customizados em self.splitters (dict correto do River)
        #    Feita ANTES do super().learn_one para que update_splitters use os nossos.
        nom_attrs = tree.nominal_attributes if (tree and hasattr(tree, 'nominal_attributes')) else []
        if self.splitters is None:
            self.splitters = {}
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

        # 4. Delega ao River com o parâmetro correto (w, não sample_weight)
        super().learn_one(x_subset, y, w=w, tree=tree)

# =============================================================================
# 2. CLASSES DE NÓS CONCRETAS (MC, NB, NBA)
# =============================================================================
# Precisamos combinar o Mixin com os tipos de nós do River para suportar
# Majority Class (MC), Naive Bayes (NB) e Naive Bayes Adaptive (NBA)

class ARTELeafMajorityClass(RandomSubspaceNodeMixin, LeafMajorityClass):
    """No Majority Class com Subespaco Aleatorio."""
    pass

class ARTELeafNaiveBayes(RandomSubspaceNodeMixin, LeafNaiveBayes):
    """No Naive Bayes com Subespaco Aleatorio."""
    pass

class ARTELeafNaiveBayesAdaptive(RandomSubspaceNodeMixin, LeafNaiveBayesAdaptive):
    """No Naive Bayes Adaptive com Subespaco Aleatorio."""
    pass
