import random
import numbers
import numpy as np
from river.tree.splitter import GaussianSplitter
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
        # 1. Atualiza min/max globais (Lógica do getSplitPointSuggestions do Java)
        if att_val < self._min_value: self._min_value = att_val
        if att_val > self._max_value: self._max_value = att_val

        # 2. Mantém a atualização das Gaussianas por classe (super)
        super().update(att_val, target_val, sample_weight)

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        # Equivalente ao getSplitPointSuggestions() + loop de verificação do Java

        # Se não temos intervalo suficiente para sortear, retorna sugestão inválida
        if self._min_value >= self._max_value:
            return BranchFactory()

        # Sorteio do ponto de corte (Lógica Java: (rand * (max - min)) + min)
        split_value = self.rng.uniform(self._min_value, self._max_value)

        # Usa o método do GaussianSplitter que lida corretamente com edge cases
        # (zero variância, split fora do range observado por classe)
        post_split_dist = self._class_dists_from_binary_split(split_value)

        # Calcula o mérito (Information Gain / Gini) deste split único
        merit = criterion.merit_of_split(pre_split_dist, post_split_dist)

        return BranchFactory(merit, att_idx, split_value, post_split_dist)
