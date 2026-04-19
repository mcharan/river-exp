"""
hetero_bagging.py — Runner para HeterogeneousOnlineBagging.

Ensemble heterogeneo de MLPs com Online Bagging e deteccao de drift
direcional por membro (ADWINChangeDetector).

Uso:
    python experiments/neural_arte/hetero_bagging.py --dataset electricity --composition abc
    python experiments/neural_arte/hetero_bagging.py --dataset sea_a --composition abc_proj --gpu 0
    python experiments/neural_arte/hetero_bagging.py --dataset covtype --no_drift
"""

import sys
import os

# Early-parse --gpu from sys.argv to set CUDA_VISIBLE_DEVICES BEFORE torch is
# imported. Necessary because tmux new-session inherits the server environment,
# which may have CUDA_VISIBLE_DEVICES=0 even when the orchestrator passes --gpu -1.
_early_gpu = None
for _i, _arg in enumerate(sys.argv):
    if _arg == '--gpu' and _i + 1 < len(sys.argv):
        try:
            _early_gpu = int(sys.argv[_i + 1])
        except ValueError:
            pass
        break

if (_early_gpu is not None and _early_gpu < 0) or \
        os.environ.get('CUDA_VISIBLE_DEVICES', '') == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import numpy as np
import time
import argparse
import psutil
import datetime

from river import metrics

from src.neural_arte.neural_arte import (
    FastIncrementalScaler,
    get_dataset_universal,
    apply_one_hot_encoding,
    log_results_to_csv,
    NoDriftDetector,
    DATASETS_PATH as _DEFAULT_DATASETS_PATH,
)
from src.neural_arte.heterogeneous_bagging import HeterogeneousOnlineBagging, COMPOSITIONS
from src.shared.metrics import KappaM


def main(dataset, seed, n_models, lambda_val, window_size, composition,
         delta, datasets_path, datasets_path_fallback, device, use_drift):

    import src.neural_arte.neural_arte as _nn_mod
    if datasets_path:
        _nn_mod.DATASETS_PATH = datasets_path

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- HeteroBagging | composition={composition} | device={device} ---")

    # ------------------------------------------------------------------
    # 1. Carrega dados
    # ------------------------------------------------------------------
    try:
        X_all, y_all, n_feat_raw, n_classes, nom_indices = get_dataset_universal(
            dataset, seed=seed, datasets_path_fallback=datasets_path_fallback
        )
    except Exception as e:
        print(f"Erro carregando dataset: {e}")
        return

    if nom_indices:
        X_all = apply_one_hot_encoding(X_all, nom_indices)

    n_feat = X_all.shape[1]
    print(f"Dataset: {dataset} | Inst: {len(X_all)} | Feat: {n_feat} | Classes: {n_classes}")

    # ------------------------------------------------------------------
    # 2. Pre-escalonamento + tensorização para GPU
    # ------------------------------------------------------------------
    print("Pre-escalonando dataset...")
    X_scaled_all = np.zeros_like(X_all, dtype=np.float32)
    scaler_pre = FastIncrementalScaler(n_feat)
    for i in range(len(X_all)):
        scaler_pre.learn_one(X_all[i])
        X_scaled_all[i] = scaler_pre.transform_one(X_all[i])
    X_gpu = torch.tensor(X_scaled_all, device=device, dtype=torch.float32)
    print("Pronto. Construindo ensemble...")

    # ------------------------------------------------------------------
    # 3. Constroi o modelo
    # ------------------------------------------------------------------
    torch.manual_seed(seed)

    _delta = delta if use_drift else 1e9  # delta enorme ≈ drift nunca dispara
    model = HeterogeneousOnlineBagging(
        n_features=n_feat,
        n_classes=n_classes,
        composition=composition,
        n_models=n_models,
        lambd=lambda_val,
        window_size=window_size,
        seed=seed,
        delta=_delta,
        device=device,
    )

    drift_tag = 'adwin_dir' if use_drift else 'nodrift'
    print(f"n_models: {n_models} | Drift: {drift_tag}")

    # ------------------------------------------------------------------
    # 4. Métricas + arquivo de saída
    # ------------------------------------------------------------------
    metric_acc     = metrics.Accuracy()
    metric_kappa   = metrics.CohenKappa()
    metric_kappa_m = KappaM()
    metric_gmean   = metrics.GeometricMean()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        f"results/neural/HeteroBagging_{dataset}_{composition}"
        f"_{drift_tag}_s{seed}_{timestamp}.csv"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Salvando em: {output_file}")

    run_id = f"hetero_bagging_{composition}_{drift_tag}"
    latencies = []
    start_total = time.time()
    log_interval = 2000

    def save_snapshot(current_count, force=False):
        ram  = psutil.Process().memory_info().rss / (1024 * 1024)
        vram = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

        slice_size = min(len(latencies), 2000)
        avg_lat = sum(latencies[-slice_size:]) / slice_size if latencies else 0.0

        stats_dict = {
            "Run_ID":      run_id,
            "Time":        datetime.datetime.now().strftime("%H:%M:%S"),
            "Instancia":   current_count,
            "Dataset":     dataset,
            "Accuracy":    metric_acc.get(),
            "Kappa":       metric_kappa.get(),
            "KappaM":      metric_kappa_m.get(),
            "GMean":       metric_gmean.get(),
            "Latencia_ms": avg_lat,
            "Drifts":      model.total_drifts,
            "RAM_MB":      ram,
            "VRAM_MB":     vram,
        }
        log_results_to_csv(output_file, stats_dict)
        if force or current_count % 10000 == 0:
            print(
                f"[{dataset}] Inst: {current_count} | Acc: {metric_acc.get():.2%} | "
                f"Kappa: {metric_kappa.get():.2f} | Drifts: {model.total_drifts} | "
                f"RAM: {ram:.0f}MB"
            )

    # ------------------------------------------------------------------
    # 5. Loop prequencial
    # ------------------------------------------------------------------
    print("Iniciando loop prequencial...")
    for count in range(len(X_all)):
        y = int(y_all[count])

        t0 = time.perf_counter()
        x_tensor = X_gpu[count]
        y_pred = model.predict_one(x_tensor)
        t_pred = time.perf_counter() - t0

        metric_acc.update(y, y_pred)
        metric_kappa.update(y, y_pred)
        metric_kappa_m.update(y, y_pred)
        metric_gmean.update(y, y_pred)

        t1 = time.perf_counter()
        model.learn_one(x_tensor, y)
        t_learn = time.perf_counter() - t1

        latencies.append((t_pred + t_learn) * 1000)

        if (count + 1) % log_interval == 0:
            save_snapshot(count + 1)

    if len(X_all) % log_interval != 0:
        save_snapshot(len(X_all), force=True)

    print(
        f"Fim {dataset}. Tempo Total: {(time.time() - start_total):.1f}s | "
        f"Acc: {metric_acc.get():.2%} | Drifts: {model.total_drifts}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runner para HeterogeneousOnlineBagging com ADWIN direcional.'
    )
    parser.add_argument('--dataset',            type=str, required=True)
    parser.add_argument('--composition',        type=str, default='abc',
                        choices=list(COMPOSITIONS.keys()),
                        help='Composicao do ensemble (abc | abc_proj | heterogeneous).')
    parser.add_argument('--seed',               type=int, default=123456789)
    parser.add_argument('--n_models',           type=int, default=30)
    parser.add_argument('--lambda_val',         type=int, default=6)
    parser.add_argument('--window',             type=int, default=500)
    parser.add_argument('--delta',              type=float, default=0.001)
    parser.add_argument('--datasets_path',      type=str, default=None)
    parser.add_argument('--datasets_path_real', type=str, default=None,
                        help='Caminho fallback (pasta full) para datasets reais.')
    parser.add_argument('--gpu',                type=int, default=0,
                        help='Indice da GPU (0, 1, ...) ou -1 para CPU.')
    parser.add_argument('--no_drift',           action='store_true',
                        help='Desativa o detector de drift.')

    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        _device = f'cuda:{args.gpu}'
    elif args.gpu == -1:
        _device = 'cpu'
    else:
        _device = None

    main(
        dataset=args.dataset,
        seed=args.seed,
        n_models=args.n_models,
        lambda_val=args.lambda_val,
        window_size=args.window,
        composition=args.composition,
        delta=args.delta,
        datasets_path=args.datasets_path,
        datasets_path_fallback=args.datasets_path_real,
        device=_device,
        use_drift=not args.no_drift,
    )
