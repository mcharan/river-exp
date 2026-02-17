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
import argparse
import random
import numpy as np
import pandas as pd
from scipy.io import arff
from metrics import KappaM
from ARTE import ARTE
from utils import log_results_to_csv, get_dataset_universal
from river import base, tree, drift, utils, stats, metrics, datasets
from river.datasets import synth
from river.tree.splitter import GaussianSplitter
from river.tree.split_criterion import VarianceReductionSplitCriterion
from river.tree.utils import BranchFactory



# In[2]:


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
    metric_kappa_m = KappaM()
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
        metric_kappa_m.update(y, y_pred)
        metric_gmean.update(y, y_pred)
        
        # 3. Learn
        t1 = time.perf_counter()
        model.learn_one(x_dict, y)
        dur_learn = time.perf_counter() - t1
        
        latencies.append((dur_pred + dur_learn) * 1000) # ms

        # 4. Logs (a cada 2k ou 10k dependendo do tamanho)
        is_last_instance = (count + 1) == len(X_all)
        log_interval = 2000
        if (count + 1) % log_interval == 0 or is_last_instance:
            ram = psutil.Process().memory_info().rss / (1024 * 1024)

            # Evita divisão por zero se o dataset for minúsculo e latencies estiver vazia
            avg_lat = 0.0
            if latencies:
                # Pega as últimas X latências ou todas se tiver menos que o intervalo
                slice_size = min(len(latencies), log_interval)
                avg_lat = sum(latencies[-slice_size:]) / slice_size
            
            stats_dict = {
                "Run_ID": run_id,
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Instancia": count + 1,
                "Dataset": dataset,
                "Accuracy": metric_acc.get(),
                "Kappa": metric_kappa.get(),
                "Kappa_M": metric_kappa_m.get(),
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



# In[ ]:


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




