import random
from river import tree

from .nodes import ARTELeafMajorityClass, ARTELeafNaiveBayes, ARTELeafNaiveBayesAdaptive


class ARTEHoeffdingTree(tree.HoeffdingTreeClassifier):
    """
    Port do ARTEHoeffdingTree.java para River.

    Parâmetros:
        subspace_size (int): O parâmetro 'k'. Define o número de features por nó.
                             Se negativo, usa (Total - k).
        seed (int): Semente aleatória para a seleção de features.
    """
    def __init__(self, subspace_size=2, seed=None, **kwargs):
        # O artigo diz: "pruning in random forests reduces variability".
        # Portanto, forçamos remove_poor_attrs=False (desativa poda de atributos ruins).
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

        # stats, depth, splitter são requeridos pelo HTLeaf (passados via **kwargs ao Mixin)
        return node_cls(
            subspace_size=self.subspace_size,
            rng=self._rng,
            stats=initial_stats,
            depth=depth,
            splitter=self.splitter,
        )
