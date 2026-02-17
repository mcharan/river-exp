import random
import collections
import statistics
import numpy as np
from river import base, tree, drift, utils, stats, metrics
from river.tree.splitter import Splitter, GaussianSplitter
from river.tree.nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
# Tenta importar as classes de folha do local correto

# except ImportError:
#     # Fallback caso a estrutura de importacao varie
#     from river.tree.nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
    
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
        
        # Se não temos intervalo suficiente para sortear, retorna None
        if self._min_value >= self._max_value:
            return None

        # Sorteio do ponto de corte (Lógica Java: (rand * (max - min)) + min)
        split_value = self.rng.uniform(self._min_value, self._max_value)
        
        # Cria a sugestão de ramo baseada nesse único ponto
        # O River precisa calcular a "post_split_dist" (distribuição das classes esq/dir)
        # O método abaixo estima isso usando as gaussianas internas
        post_split_dist = self.cond_proba(split_value)
        
        # Calcula o mérito (Information Gain / Gini) deste split único
        merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
        
        # Retorna a sugestão empacotada (BranchFactory cria o objeto AttributeSplitSuggestion)
        return BranchFactory(
            merit=merit,
            feature=att_idx,
            operator='<', # NumericBinaryTest
            value=split_value,
            numerical_feature=True,
            multiway_split=False
        )

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

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        # 1. Lógica de Seleção de Subespaço (Igual à sua anterior)
        if self.selected_features is None:
            all_features = list(x.keys())
            n_features = len(all_features)
            k = self.subspace_size
            if k < 0: k = n_features + k
            k = max(1, min(k, n_features))
            self.selected_features = self.rng.sample(all_features, k)

        # 2. Filtragem de features
        x_subset = {key: x[key] for key in self.selected_features if key in x}

        # 3. [NOVO] Injeção do ARTEGaussianSplitter
        # Antes de chamar o super().learn_one, garantimos que os observadores
        # para as features numéricas sejam da nossa classe customizada.
        for att_id, att_val in x_subset.items():
            # Se o atributo ainda não tem um observador (Splitter) criado
            if att_id not in self.stats:
                # Verifica se é nominal (via lista de nominais da árvore se disponível ou heurística)
                # O River geralmente trata strings como nominais automaticamente, mas aqui
                # vamos focar em garantir que NUMÉRICOS usem nosso Splitter.
                is_numeric = isinstance(att_val, (int, float)) and not isinstance(att_val, bool)
                
                # Se a árvore tiver lista de nominais explícita, usamos ela para segurança
                if tree is not None and hasattr(tree, 'nominal_attributes') and tree.nominal_attributes:
                     if att_id in tree.nominal_attributes:
                         is_numeric = False

                if is_numeric:
                    # Injeta nosso Splitter com o RNG do nó
                    self.stats[att_id] = ARTEGaussianSplitter(rng=self.rng)
        
        # 4. Passa para o River processar a atualização (vai usar o splitter que acabamos de criar)
        super().learn_one(x_subset, y, sample_weight=sample_weight, tree=tree)

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
        
    def _new_learning_node(self, initial_stats=None, parent=None):
        """
        Sobrescreve a criação de nós para injetar nós customizados
        que suportam Random Subspace.
        """
        # Define qual classe de nó usar baseado na configuração da folha
        if self.leaf_prediction == 'mc':
            node_cls = ARTELeafMajorityClass
        elif self.leaf_prediction == 'nb':
            node_cls = ARTELeafNaiveBayes
        elif self.leaf_prediction == 'nba':
            node_cls = ARTELeafNaiveBayesAdaptive
        else:
            node_cls = ARTELeafMajorityClass

        # Retorna o nó instanciado com o subspace_size e o gerador aleatório
        return node_cls(
            subspace_size=self.subspace_size,
            rng=self._rng,
            initial_stats=initial_stats
        )

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
        n_rejections: int = 5,
        seed: int = 1,
        k_min: int = 2
    ):

        self.n_features = n_features
        self.nominal_attributes = nominal_attributes or [] # Lista de índices
        self.n_models = n_models
        self.lambd = lambd
        self.drift_detector = drift_detector or drift.ADWIN(delta=1e-3)
        self.window_size = window_size
        self.n_rejections = n_rejections
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
                grace_period=100,
                delta=0.01
                # remove_poor_attrs já é forçado para False dentro da classe
            )
            
            m = {
                'model': tree_model,
                'detector': self.drift_detector.clone(),
                'untrained_counts': collections.defaultdict(int),
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
            # Predição para controle de erro e lógica de rejeição
            y_pred = m['model'].predict_one(x)
            correct = (y == y_pred)
            
            # Estratégia de Regularização Adaptativa:
            # Para evitar que domínios com ruído dominem, treina no erro
            # ou após N rejeições (acertos)
            will_train = not correct
            
            if correct:
                m['untrained_counts'][y] += 1
                if self.n_rejections > 0 and m['untrained_counts'][y] >= self.n_rejections:
                    m['untrained_counts'][y] = 0
                    will_train = True
            
            if will_train:
                # Online Bagging via Poisson
                k = self._rng.poisson(self.lambd)
                if k > 0:
                    for _ in range(k):
                        m['model'].learn_one(x, y)
                        m['instances_trained'] += 1
            
            # Detecção de Drift individual
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
            grace_period=100,
            delta=0.01
        )
        m['detector'] = self.drift_detector.clone()
        m['untrained_counts'].clear()
        m['window_acc'] = utils.Rolling(stats.Mean(), window_size=self.window_size)

    @property
    def total_drifts(self):
        return self._total_drifts

