#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import sys
import time
import uuid
import datetime
import psutil
import statistics
import collections
import argparse
import numpy as np
import pandas as pd
from scipy.io import arff
from river import base, tree, drift, utils, stats, metrics, datasets
from river.datasets import synth


# In[9]:


class ARTE(base.Ensemble, base.Classifier):
    """Adaptive Random Tree Ensemble (ARTE) portado do MOA.
    
    Algoritmo adaptativo para fluxos de dados evolutivos de Paim e Enembreck.
    """

    def __init__(
        self,
        model: base.Classifier = None,
        n_models: int = 100,
        lambd: float = 6.0,
        drift_detector: base.DriftDetector = None,
        window_size: int = 1000,
        n_rejections: int = 5,
        seed: int = 1
    ):
        # O modelo base sugerido no original é a ARFHoeffdingTree
        # No River, usamos HoeffdingTreeClassifier como base
        self.model = model or tree.HoeffdingTreeClassifier()
        self.n_models = n_models
        self.lambd = lambd
        self.drift_detector = drift_detector or drift.ADWIN(delta=1e-3)
        self.window_size = window_size
        self.n_rejections = n_rejections
        self.seed = seed
        self._rng = np.random.RandomState(self.seed)
        
        # Inicialização dos membros conforme a estrutura AREBaseLearner do original
        self._ensemble_members = []
        for i in range(self.n_models):
            m = {
                'model': self.model.clone(),
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
        """Reinicia o modelo e estatísticas após detecção de mudança."""
        m['model'] = self.model.clone()
        m['detector'] = self.drift_detector.clone()
        m['untrained_counts'].clear()
        m['window_acc'] = utils.Rolling(stats.Mean(), window_size=self.window_size)

    @property
    def total_drifts(self):
        return self._total_drifts


# In[11]:


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
        
        if not os.path.exists(path):
            # Fallback para datasets embutidos do River se não tiver o arquivo
            if name == 'electricity': return _load_river_dataset(datasets.Elec2())
            if name == 'shuttle': return _load_river_dataset(datasets.Shuttle())
            if name == 'covtype': return _load_river_dataset(datasets.Covertype())
            raise FileNotFoundError(f"Arquivo {filename} não encontrado e sem fallback.")

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

        # Para Árvores, NÃO precisamos de One-Hot Encoding global.
        # O Hoeffding Tree lida bem com numéricos. 
        # Categoricos devem ser passados como dict.
        # Aqui retornamos numpy, depois convertemos para dict no loop.
        
        return X.values, y.values, X.shape[1], len(np.unique(y))
    
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
        return X_np, y_np, X_np.shape[1], len(np.unique(y_np))

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
    

    


# In[13]:


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
        X_all, y_all, n_feat, n_classes = get_dataset_universal(dataset, seed=seed)
    except Exception as e:
        print(f"Erro carregando {dataset}: {e}")
        return

    print(f"Dataset: {dataset} | Inst: {len(X_all)} | Feat: {n_feat} | Classes: {n_classes}")

    # 2. Setup do Modelo
    # ARTE Original usa Hoeffding Tree e n_models=50-100
    model = ARTE(
        n_models=n_models,
        lambd=lambda_val,
        window_size=window_size, # W=500 conforme paper
        drift_detector=drift.ADWIN(delta=0.001),
        seed=seed
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
            'agrawal_g',
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
            main_arte_cpu(dataset=ds, seed=123456789, n_models=50, window_size=500)
    


# In[ ]:




