"""
GNN-ARTE: NeuralARTE com Meta-GNN para Agregação do Ensemble
=============================================================
Baseado em: NeuralARTE/neural_arte.py
Diferencial: substitui votação majoritária por MetaGNNAggregator.

Parâmetros adicionais vs. NeuralARTE:
  --gnn_hidden  : dimensão oculta da GNN (padrão 64)
  --gnn_update  : instâncias entre atualizações do Meta-GNN (padrão 10)
  --gnn_heads   : número de cabeças de atenção GAT (padrão 4)
  --graph_type  : 'full' | 'knn' (padrão 'full')
  --no_metagnn  : desativa Meta-GNN (usa votação majoritária — baseline)
"""

import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import collections
import time
import csv
import os
import sys
import argparse
import psutil
import datetime

from scipy.io import arff
from river import stats, drift, metrics

# --- imports locais ----------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.shared.metrics import KappaM

from src.neural_arte.neural_arte import (
    get_dataset_universal,
    apply_one_hot_encoding,
    log_results_to_csv,
    FastIncrementalScaler,
    FlexibleNeuralNetwork,
    DATASETS_PATH as _DEFAULT_DATASETS_PATH,
)

from deep_river import classification

from src.gnn.ensemble_gnn import MetaGNNAggregator, HAS_PYG

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
DATASETS_PATH = _DEFAULT_DATASETS_PATH


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================
def main_gnn_arte(dataset: str, seed: int, n_models: int, lambda_val: int,
                  window_size: int, datasets_path: str = None,
                  device: str = None, batch_size: int = 32,
                  gnn_hidden: int = 64, gnn_update: int = 10,
                  gnn_heads: int = 4, graph_type: str = "full",
                  use_metagnn: bool = True):

    global DATASETS_PATH
    if datasets_path:
        import src.neural_arte.neural_arte as _na
        _na.DATASETS_PATH = datasets_path
        DATASETS_PATH = datasets_path

    # --- dispositivo ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"[GNN-ARTE] Dispositivo: {dev}")
    if "cuda" in str(dev):
        print(f"  GPU: {torch.cuda.get_device_name(dev)}")

    # --- dados ---
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_all, y_all, n_feat, n_classes, nominal_idx = get_dataset_universal(
        dataset, seed=seed)
    X_all = apply_one_hot_encoding(X_all, nominal_idx)
    n_feat = X_all.shape[1]
    print(f"   >> {len(X_all)} instâncias | {n_feat} features | {n_classes} classes")

    # --- pré-tensorização ---
    print("Pré-tensorizando dataset...")
    X_all = X_all.astype(np.float32)
    scaler_pre = FastIncrementalScaler(n_feat)
    X_scaled_all = np.zeros_like(X_all)
    for i in range(len(X_all)):
        scaler_pre.learn_one(X_all[i])
        X_scaled_all[i] = scaler_pre.transform_one(X_all[i])
    X_gpu = torch.tensor(X_scaled_all, device=dev, dtype=torch.float32)
    y_all_np = y_all.astype(np.int64)
    print("   >> Pré-tensorização completa.")

    # --- ensemble de modelos base (MLPs via deep-river) ---
    rng = np.random.RandomState(seed)
    ensemble = []
    for _ in range(n_models):
        model = FlexibleNeuralNetwork(n_feat, n_classes, [64, 32])
        clf = classification.Classifier(
            module=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_fn=optim.Adam,
            lr=0.005,
            device=dev,
        )
        ensemble.append(clf)

    # --- detectores de drift (um por modelo) ---
    detectors = [drift.ADWIN(delta=1e-3) for _ in range(n_models)]
    last_drift = [0] * n_models        # instância do último drift de cada modelo
    last_loss  = [0.0] * n_models      # último loss de cada modelo
    drift_flag = [0] * n_models        # 1 se driftou nas últimas 200 inst.

    # --- Meta-GNN ---
    aggregator = None
    if use_metagnn:
        if not HAS_PYG:
            print("[GNN-ARTE] torch_geometric ausente — usando fallback MLP como Meta-GNN")
        aggregator = MetaGNNAggregator(
            n_models=n_models,
            n_classes=n_classes,
            device=dev,
            hidden_dim=gnn_hidden,
            n_heads=gnn_heads,
            lr=1e-3,
            update_every=gnn_update,
            graph_type=graph_type,
        )
        print(f"[GNN-ARTE] MetaGNN ativado (hidden={gnn_hidden}, update_every={gnn_update}, "
              f"heads={gnn_heads}, graph={graph_type})")
    else:
        print("[GNN-ARTE] Votação majoritária (baseline sem MetaGNN)")

    # --- métricas ---
    metric_acc   = metrics.Accuracy()
    metric_kappa = metrics.CohenKappa()
    metric_kappa_m = KappaM()
    total_drifts = 0

    # --- saída CSV ---
    if not use_metagnn:
        tag_gnn = "baseline"
    elif graph_type == "knn":
        tag_gnn = "metagnn_knn"
    else:
        tag_gnn = "metagnn"
    os.makedirs("results/gnn", exist_ok=True)
    csv_path = f"results/gnn/{dataset}_{tag_gnn}_s{seed}.csv"

    n_total = len(X_all)
    log_interval = 2000

    print(f"[GNN-ARTE] Iniciando loop | {n_total} instâncias | log a cada {log_interval}")
    t_start = time.time()

    for i in range(n_total):
        x_t  = X_gpu[i]
        y_t  = int(y_all_np[i])
        x_np = X_scaled_all[i]

        # --- Poisson(lambda) por modelo ---
        weights = rng.poisson(lambda_val, n_models).astype(float)

        # --- coleta predições de cada modelo ---
        x_dict = {j: float(x_np[j]) for j in range(n_feat)}
        probas_list = []
        preds_list  = []

        for m_idx, clf in enumerate(ensemble):
            proba = clf.predict_proba_one(x_dict)
            probas_list.append([proba.get(c, 0.0) for c in range(n_classes)])
            pred_m = max(proba, key=proba.get) if proba else 0
            preds_list.append(pred_m)

            # drift detection
            correct = 1 if pred_m == y_t else 0
            detectors[m_idx].update(1 - correct)
            if detectors[m_idx].drift_detected:
                model_new = FlexibleNeuralNetwork(n_feat, n_classes, [64, 32])
                clf_new = classification.Classifier(
                    module=model_new, loss_fn=nn.CrossEntropyLoss(),
                    optimizer_fn=optim.Adam, lr=0.005, device=dev)
                ensemble[m_idx] = clf_new
                detectors[m_idx] = drift.ADWIN(delta=1e-3)
                total_drifts += 1
                last_drift[m_idx] = i
                drift_flag[m_idx] = 1
            else:
                if drift_flag[m_idx] and (i - last_drift[m_idx]) > 200:
                    drift_flag[m_idx] = 0

        # --- agregação ---
        # last_loss usa a perda da instância ANTERIOR (sem leakage do label atual)
        # warm-up: usa votação majoritária até o GNN ter treinado o suficiente
        t0 = time.time()
        if aggregator is not None and i >= aggregator.update_every * 10:
            node_feats = aggregator.collect_node_features(
                probas_list, last_loss, drift_flag)
            y_pred = aggregator.predict(node_feats)
        else:
            node_feats = aggregator.collect_node_features(
                probas_list, last_loss, drift_flag) if aggregator is not None else None
            votes = collections.Counter(preds_list)
            y_pred = votes.most_common(1)[0][0]
        lat_ms = (time.time() - t0) * 1000

        # --- métricas ---
        metric_acc.update(y_t, y_pred)
        metric_kappa.update(y_t, y_pred)
        metric_kappa_m.update(y_t, y_pred)

        # --- treino online (Poisson) ---
        # deep_river Classifier não aceita peso diretamente:
        # simula Online Bagging chamando learn_one k=Poisson(lambda) vezes
        for m_idx, clf in enumerate(ensemble):
            for _ in range(int(weights[m_idx])):
                clf.learn_one(x_dict, y_t)

        # --- atualiza last_loss com label revelado (para próxima instância) ---
        for m_idx in range(n_models):
            p_true = probas_list[m_idx][y_t] if y_t < n_classes else 1e-9
            last_loss[m_idx] = -np.log(max(p_true, 1e-9))

        # --- treino Meta-GNN ---
        if aggregator is not None and node_feats is not None:
            aggregator.update(node_feats, y_t)

        # --- log periódico ---
        if (i + 1) % log_interval == 0 or i == n_total - 1:
            elapsed = time.time() - t_start
            ram_mb = psutil.Process().memory_info().rss / 1e6
            row = {
                "Dataset":      dataset,
                "Instance":     i + 1,
                "Accuracy":     metric_acc.get(),
                "Kappa":        metric_kappa.get(),
                "KappaM":       metric_kappa_m.get(),
                "Drifts":       total_drifts,
                "Latencia_ms":  lat_ms,
                "RAM_MB":       ram_mb,
                "Time":         datetime.datetime.now().strftime("%H:%M:%S"),
            }
            log_results_to_csv(csv_path, row)

            if (i + 1) % (log_interval * 25) == 0 or i == n_total - 1:
                print(f"  [{i+1:>8}/{n_total}] "
                      f"Acc={metric_acc.get()*100:.2f}% | "
                      f"K={metric_kappa.get():.3f} | "
                      f"Drifts={total_drifts} | "
                      f"Lat={lat_ms:.2f}ms | "
                      f"RAM={ram_mb:.0f}MB | "
                      f"t={elapsed/60:.1f}min")

    print(f"\n=== {dataset.upper()} [{tag_gnn}] CONCLUÍDO ===")
    print(f"  Acc={metric_acc.get()*100:.2f}% | Kappa={metric_kappa.get():.3f} | "
          f"KappaM={metric_kappa_m.get():.3f} | Drifts={total_drifts} | "
          f"Tempo={(time.time()-t_start)/60:.1f}min")
    print(f"  CSV: {csv_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-ARTE: NeuralARTE com Meta-GNN")
    parser.add_argument("--dataset",       required=True)
    parser.add_argument("--seed",          type=int,   default=123456789)
    parser.add_argument("--n_models",      type=int,   default=30)
    parser.add_argument("--lambda_val",    type=int,   default=6)
    parser.add_argument("--window",        type=int,   default=500)
    parser.add_argument("--datasets_path", type=str,   default=None)
    parser.add_argument("--device",        type=str,   default=None,
                        help="cuda | cuda:0 | cuda:1 | cpu")
    parser.add_argument("--gnn_hidden",    type=int,   default=64)
    parser.add_argument("--gnn_update",    type=int,   default=10,
                        help="Instâncias entre atualizações do Meta-GNN")
    parser.add_argument("--gnn_heads",     type=int,   default=4)
    parser.add_argument("--graph_type",    type=str,   default="full",
                        choices=["full", "knn"])
    parser.add_argument("--no_metagnn",    action="store_true",
                        help="Usa votação majoritária (baseline)")
    args = parser.parse_args()

    main_gnn_arte(
        dataset=args.dataset,
        seed=args.seed,
        n_models=args.n_models,
        lambda_val=args.lambda_val,
        window_size=args.window,
        datasets_path=args.datasets_path,
        device=args.device,
        gnn_hidden=args.gnn_hidden,
        gnn_update=args.gnn_update,
        gnn_heads=args.gnn_heads,
        graph_type=args.graph_type,
        use_metagnn=not args.no_metagnn,
    )
