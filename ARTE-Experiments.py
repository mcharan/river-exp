#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import uuid
import datetime
import psutil
import statistics
import collections
import argparse
import random
import numpy as np
import pandas as pd
from scipy.io import arff
from river import base, tree, drift, utils, stats, metrics, datasets
from river.datasets import synth
# Tenta importar as classes de folha do local correto
try:
    # Caminho mais comum em versoes recentes
    from river.tree.nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
except ImportError:
    # Fallback caso a estrutura de importacao varie
    from river.tree.nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive


# In[2]:


# =============================================================================
# 1. MIXIN DE SUBESPAÇO ALEATÓRIO (Lógica central do RandomLearningNode.java)
# =============================================================================
class RandomSubspaceNodeMixin:
    """
    Mixin que implementa a lógica de seleção de subespaço aleatório por nó.
    Equivalente ao RandomLearningNode do Java.
    """
    def __init__(self, subspace_size, rng, **kwargs):
        super().__init__(**kwargs)
        self.subspace_size = subspace_size
        self.rng = rng
        self.selected_features = None  # Lista de atributos selecionados (int[] listAttributes no Java)

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        # Seleção Lazy (na primeira vez que vê uma instância), igual ao Java
        if self.selected_features is None:
            all_features = list(x.keys())
            n_features = len(all_features)
            
            k = self.subspace_size
            
            # Lógica do Java: "Negative values = #features - k"
            if k < 0:
                k = n_features + k
            
            # Garante limites
            k = max(1, min(k, n_features))
            
            # Sorteio sem reposição (implementa o loop while/check unique do Java)
            self.selected_features = self.rng.sample(all_features, k)

        # Filtra o dicionário x mantendo APENAS as features selecionadas
        # Isso garante que as estatísticas (Gaussianas/Histogramas) só sejam criadas para essas features.
        # Equivalente ao loop "for (int j = 0; j < this.numAttributes; j++)" do Java
        x_subset = {key: x[key] for key in self.selected_features if key in x}

        # Passa o x filtrado para a lógica original do River (update stats)
        super().learn_one(x_subset, y, sample_weight=sample_weight, tree=tree)

    def best_split_suggestions(self, criterion, tree):
        # Como filtramos o x no learn_one, o super().best_split_suggestions 
        # só vai considerar os atributos que existem nas estatísticas internas.
        return super().best_split_suggestions(criterion, tree)

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


# In[3]:


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


# In[4]:


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


# In[5]:


def get_dataset_universal(dataset_name, seed=42, n_synthetic=1000000):
    """
    Suporta Datasets Reais (ARFF) e Sintéticos com Drift Simulado (Blocos).
    """
    name = dataset_name.lower()
    # Ajuste este caminho para o seu ambiente
    paim_path = "/home/charan/moa/aldopaim/AdaptiveRandomTreeEnsemble/datasets" 
    
    # Mapeamento Tabela 14 - Reais
    real_files = {
        'electricity': 'elecNormNew.arff', # ou electricity.arff
        'elec2': 'elecNormNew.arff',
        'airlines': 'airlines.arff',
        'covtype': 'covtypeNorm.arff',   # 581k inst
        'gassensor': 'gassensor.arff',   # 13k
        'gmsc': 'gmsc.arff',             # 150k
        'keystroke': 'keystroke.arff',   # 1.6k
        'outdoor': 'outdoor.arff',       # 4k
        'ozone': 'ozone.arff',           # 2.5k
        'rialto': 'rialto.arff',         # 82k
        'shuttle': 'shuttle.arff',       # 58k
        'noaa': 'NOAA.arff'
    }

    # --- 1. REAIS ---
    if name in real_files:
        filename = real_files[name]
        path = os.path.join(paim_path, filename)
        
        print(f"Carregando ARFF Real: {filename}...")
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        
        # Decode bytes
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')

        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Tratamento de Labels (String/Bytes -> Int)
        # Tenta converter para numérico, se falhar, usa fatorização
        try:
            y = pd.to_numeric(y)
        except:
            # Se for string (ex: 'UP', 'Class1'), converte para categorical codes
            y = pd.Categorical(y).codes

        # --- DETECÇÃO DE NOMINAIS ---
        nominal_attributes = []
        X_final = X.copy()

        for idx, col in enumerate(X.columns):
            # Se for object ou category, ou se tiver poucos valores únicos (heurística opcional)
            if X[col].dtype == object or X[col].dtype.name == 'category':
                nominal_attributes.append(idx)
                # Converte string -> int (Label Encoding) para o River processar rápido
                # Mas avisaremos a árvore que esse int é uma categoria!
                X_final[col] = pd.Categorical(X[col]).codes
            else:
                # Garante float para numéricos
                X_final[col] = pd.to_numeric(X_final[col], errors='coerce').fillna(0.0)
        # Para Árvores, NÃO precisamos de One-Hot Encoding global.
        # O Hoeffding Tree lida bem com numéricos. 
        # Categoricos devem ser passados como dict.
        # Aqui retornamos numpy, depois convertemos para dict no loop.
        
        return X.values, y.values, X.shape[1], len(np.unique(y)), nominal_attributes
    
    # --- 2. SINTÉTICOS COM DRIFT (TABELA 14) ---
    # Lógica: Drift a cada 250.000 instâncias (total 1M)
    parts = name.split('_')
    base_name = parts[0]
    drift_type = parts[1] if len(parts) > 1 else 'n' # a=abrupt, g=gradual

    gen_list = []
    
    if base_name == 'agrawal':
        # Agrawal: 10 funções. Abrupto: troca função.
        funcs = [0, 2, 4, 6] # Escolha arbitrária de funções distintas
        for f in funcs: gen_list.append(synth.Agrawal(classification_function=f, seed=seed))
            
    elif base_name == 'led':
        # LED: Drift de ruído ou atributos irrelevantes
        gen_list = [
            synth.LED(seed=seed, noise_percentage=0.0),
            synth.LED(seed=seed, noise_percentage=0.1), # Mais ruído
            synth.LED(seed=seed, noise_percentage=0.2),
            synth.LED(seed=seed, noise_percentage=0.3)
        ]
        
    elif base_name == 'sea':
        # SEA: 4 funções baseadas em limiar
        for f in range(4): gen_list.append(synth.SEA(classification_function=f, seed=seed))
            
    elif base_name == 'mixed':
        # Mixed: Boolean functions
        # Simula drift invertendo ou mudando algo (Mixed não tem params fáceis no River,
        # usaremos seeds diferentes ou alternância se possível. 
        # O River tem synth.Mixed(). Vamos alternar seeds para simular mudança).
        gen_list = [synth.Mixed(seed=seed+i) for i in range(4)]

    elif base_name == 'rbf':
        # RBF: Centróides móveis. 
        # O artigo cita 'f' (fast) e 'm' (moderate).
        # No River, change_speed controla isso.
        speed = 0.001 if drift_type == 'f' else 0.0001
        # RBF gera drift nativamente, não precisa de blocos!
        # Retornamos gerador direto.
        gen = synth.RandomRBF(seed=seed, n_classes=5, n_features=10, n_centroids=50, change_speed=speed)
        return _gen_to_numpy(gen, n_synthetic)

    # Processamento dos Blocos (para Agrawal, SEA, LED, Mixed)
    if gen_list:
        if drift_type == 'n': # Sem drift
            return _gen_to_numpy(gen_list[0], n_synthetic)
        
        # Com drift (Abrupto/Gradual simulado por blocos)
        block_size = 250000
        print(f"Gerando Sintético {name} com Drift (4 blocos de {block_size})...")
        X_final, y_final = [], []
        
        # Garante que tem 4 geradores (cicla se precisar)
        while len(gen_list) < 4: gen_list.extend(gen_list)
        
        for i in range(4):
            print(f"   -> Bloco {i+1}...")
            X_b, y_b = _gen_to_numpy(gen_list[i], block_size, ret_meta=False)
            X_final.extend(X_b); y_final.extend(y_b)
            
        X_np = np.array(X_final)
        y_np = np.array(y_final)
        return X_np, y_np, X_np.shape[1], len(np.unique(y_np)), []

    raise ValueError(f"Dataset {name} desconhecido")

def _gen_to_numpy(gen, n, ret_meta=True):
    X, y = [], []
    for x_dict, y_val in gen.take(n):
        X.append(list(x_dict.values()))
        y.append(y_val)
    if not ret_meta: return X, y
    X_np = np.array(X); y_np = np.array(y)
    return X_np, y_np, X_np.shape[1], len(np.unique(y_np))

def _load_river_dataset(dataset):
    print(f"Usando dataset embutido do River: {dataset.__class__.__name__}")
    X, y = [], []
    for x, y_val in dataset:
        X.append(list(x.values()))
        y.append(y_val)
    # Label encoding se for string
    y_s = pd.Series(y)
    if y_s.dtype == object: y = y_s.astype('category').cat.codes.values
    else: y = np.array(y)
    X_np = np.array(X)
    return X_np, y, X_np.shape[1], len(np.unique(y))
    

    


# In[6]:


# =============================================================================
# 3. UTILS DE LOGGING
# =============================================================================
def log_results_to_csv(filename, stats_dict):
    df = pd.DataFrame([stats_dict])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


# In[ ]:


# =============================================================================
# 4. LOOP DE EXECUÇÃO (CPU)
# =============================================================================
def main_arte_cpu(dataset='airlines', seed=1, n_models=50, lambda_val=6.0, window_size=500):
    
    # 1. Carrega Dados (NumPy)
    print(f"--- Carregando {dataset} ---")
    try:
        X_all, y_all, n_feat, n_classes, nom_indices = get_dataset_universal(dataset, seed=seed)
    except Exception as e:
        print(f"Erro carregando {dataset}: {e}")
        return

    print(f"Dataset: {dataset} | Inst: {len(X_all)} | Feat: {n_feat} | Classes: {n_classes}")
    print(f"Atributos Nominais (Indices): {nom_indices}")

    # 2. Setup do Modelo
    # ARTE Original usa Hoeffding Tree e n_models=50-100
    model = ARTE(
        n_features=n_feat,
        nominal_attributes=nom_indices,
        n_models=n_models,
        lambd=lambda_val,
        window_size=window_size, # W=500 conforme paper
        drift_detector=drift.ADWIN(delta=0.001),
        seed=seed,
        k_min=2
    )

    # Métricas
    metric_acc = metrics.Accuracy()
    metric_kappa = metrics.CohenKappa()
    metric_gmean = metrics.GeometricMean() # Funciona bem para binário. Multiclasse calcula macro.

    # Setup Arquivo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    os.makedirs("results/original", exist_ok=True)
    output_file = f"results/original/ARTE_CPU_{dataset}_s{seed}_{timestamp}.csv"

    print(f"Iniciando Execução CPU | Run ID: {run_id}")
    print(f"Salvando em: {output_file}")

    start_total = time.time()
    latencies = []

    # Loop Instância a Instância (River Padrão)
    for count in range(len(X_all)):
        
        # Prepara dados (NumPy Array -> Dict para Hoeffding Tree)
        # Hoeffding Tree precisa de nomes ou índices. Usaremos índices '0', '1'...
        x_raw = X_all[count]
        y = y_all[count]
        
        # Conversão rápida para dict (gargalo do python, mas necessário para River Trees)
        x_dict = {i: x_raw[i] for i in range(n_feat)}
        
        # 1. Predict
        t0 = time.perf_counter()
        y_pred = model.predict_one(x_dict)
        dur_pred = time.perf_counter() - t0
        
        # 2. Metrics
        metric_acc.update(y, y_pred)
        metric_kappa.update(y, y_pred)
        metric_gmean.update(y, y_pred)
        
        # 3. Learn
        t1 = time.perf_counter()
        model.learn_one(x_dict, y)
        dur_learn = time.perf_counter() - t1
        
        latencies.append((dur_pred + dur_learn) * 1000) # ms

        # 4. Logs (a cada 2k ou 10k dependendo do tamanho)
        log_interval = 2000
        if (count + 1) % log_interval == 0:
            ram = psutil.Process().memory_info().rss / (1024 * 1024)
            avg_lat = sum(latencies[-log_interval:]) / log_interval
            
            stats_dict = {
                "Run_ID": run_id,
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Instancia": count + 1,
                "Dataset": dataset,
                "Accuracy": metric_acc.get(),
                "Kappa": metric_kappa.get(),
                "GMean": metric_gmean.get(),
                "Latencia_ms": avg_lat,
                "Drifts": model.total_drifts,
                "RAM_MB": ram,
                "T_Pred": dur_pred,
                "T_Learn": dur_learn
            }
            log_results_to_csv(output_file, stats_dict)
            print(f"[{dataset}] Inst: {count+1} | Acc: {metric_acc.get():.2%} | Kappa: {metric_kappa.get():.2f} | Drifts: {model.total_drifts} | RAM: {ram:.1f}MB")

    total_time = time.time() - start_total
    print(f"Fim. Tempo total: {total_time:.2f}s")
    print(f"Resultado Final - Acc: {metric_acc.get():.2%} | Kappa: {metric_kappa.get():.4f}")
    print("="*60)

# =============================================================================
# 5. EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    # Verifica se estamos no Jupyter ou Terminal
    # O Jupyter geralmente tem 'ipykernel_launcher' ou '-f' nos argumentos
    is_jupyter = any('ipykernel' in arg or '-f' in arg for arg in sys.argv)
    
    # Se rodado via linha de comando
    if not is_jupyter and len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Rodar Experimento ARTE Original CPU')
        parser.add_argument('--dataset', type=str, required=True, help='Nome do Dataset (ex: airlines, sea_a)')
        parser.add_argument('--seed', type=int, default=123456789, help='Seed aleatoria')
        parser.add_argument('--n_models', type=int, default=50, help='Numero de arvores')
        parser.add_argument('--window_size', type=int, default=500, help='Janela de acuracia')
        
        args = parser.parse_args()
        
        main_arte_cpu(
            dataset=args.dataset,
            seed=args.seed,
            n_models=args.n_models,
            lambda_val=6.0,
            window_size=args.window_size
        )
    else:
        # Exemplo de chamada para reproduzir experimentos
        # Selecione os datasets que deseja rodar:
    
        # Configuração completa para replicar a Tabela 14 do artigo original
        datasets_to_run = [
            # --- DATASETS REAIS ---
            'covtype',
            'covtype',      # 581k instancias
            'electricity',  # 45k
            'gassensor',    # 13k
            'gmsc',         # 150k
            'keystroke',    # 1.6k
            'outdoor',      # 4k
            'ozone',        # 2.5k
            'rialto',       # 82k
            'shuttle',      # 58k
            
            # --- DATASETS SINTETICOS (Com Drift) ---
            # A = Abrupt, G = Gradual (simulados por blocos na get_dataset_universal)
            
            'agrawal_a',    # 1M inst
            'agrawal_g',    # 1M inst
            'led_a',        # 1M inst
            'led_g',        # 1M inst
            'sea_a',        # 1M inst
            'sea_g',        # 1M inst
            
            # Mixed: O artigo cita Balanced e Imbalanced. 
            # Como o gerador Mixed do River é fixo, vamos usar o padrao como 'a'
            'mixed_a',      # 1M inst
            
            # RBF: O artigo cita Fast (f) e Moderate (m)
            'rbf_f',        # 1M inst (Drift rapido)
            'rbf_m'         # 1M inst (Drift moderado)
        ]
        
        # Execucao em loop
        for ds in datasets_to_run:
            # Nota: seed fixa para garantir reprodutibilidade
            main_arte_cpu(dataset=ds, seed=123456789, n_models=100, window_size=500)
    


# In[ ]:




