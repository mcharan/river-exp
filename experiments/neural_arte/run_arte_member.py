"""
run_arte_member.py — Runner para ARTESubspaceNN e ARTESoftResetNN.

Duas arquiteturas com reset por membro individual (não do ensemble inteiro):

  subspace   — ARTESubspaceNN: cada MLP usa um subconjunto aleatório de features;
               no drift, o membro é completamente reiniciado com um novo subconjunto.

  soft_reset — ARTESoftResetNN: composição diversificada (igual ao ARTELight);
               no drift, somente as últimas camadas lineares do membro são
               reinicializadas, preservando as camadas de extração de features.

Uso:
    python experiments/neural_arte/run_arte_member.py --arch subspace --dataset electricity
    python experiments/neural_arte/run_arte_member.py --arch soft_reset --dataset sea_a --composition abc
    python experiments/neural_arte/run_arte_member.py --arch soft_reset --dataset covtype --n_reset_layers 2 --no_drift
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torch import nn, optim
import numpy as np
import time
import argparse
import psutil
import datetime

from river import metrics

from src.neural_arte.neural_arte import (
    FlexibleNeuralNetwork,
    COMPOSITIONS,
    NoDriftDetector,
    FastIncrementalScaler,
    get_dataset_universal,
    apply_one_hot_encoding,
    log_results_to_csv,
    DATASETS_PATH as _DEFAULT_DATASETS_PATH,
)
from src.arte.drift_detector import ADWINChangeDetector
from src.shared.metrics import KappaM
from src.neural_arte.arte_member_reset import ARTESubspaceNN, ARTESoftResetNN
from deep_river import classification


def main(dataset, seed, n_models, lambda_val, window_size, arch,
         n_reset_layers=1, hidden_layers=None, lr=0.005, delta=0.001,
         composition='abc', datasets_path=None, datasets_path_fallback=None,
         device=None, use_drift=True):

    import src.neural_arte.neural_arte as _nn_mod
    if datasets_path:
        _nn_mod.DATASETS_PATH = datasets_path

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Iniciando arte_member (arch={arch}, device={device}) ---")

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

    # ------------------------------------------------------------------
    # 2. One-Hot Encoding
    # ------------------------------------------------------------------
    if nom_indices:
        X_all = apply_one_hot_encoding(X_all, nom_indices)

    n_feat = X_all.shape[1]
    print(f"Dataset: {dataset} | Inst: {len(X_all)} | Feat: {n_feat} | Classes: {n_classes}")

    # ------------------------------------------------------------------
    # 3. Pré-escalonamento + tensorização para GPU
    # ------------------------------------------------------------------
    print("Pré-escalonando dataset...")
    X_scaled_all = np.zeros_like(X_all, dtype=np.float32)
    scaler_pre = FastIncrementalScaler(n_feat)
    for i in range(len(X_all)):
        scaler_pre.learn_one(X_all[i])
        X_scaled_all[i] = scaler_pre.transform_one(X_all[i])
    X_gpu = torch.tensor(X_scaled_all, device=device, dtype=torch.float32)
    print("Pronto. Construindo ensemble...")

    # ------------------------------------------------------------------
    # 4. Constrói o modelo
    # ------------------------------------------------------------------
    torch.manual_seed(seed)

    if arch == 'subspace':
        _hidden = hidden_layers if hidden_layers is not None else [64]
        model = ARTESubspaceNN(
            n_features=n_feat,
            n_classes=n_classes,
            n_models=n_models,
            lambd=lambda_val,
            k_min=2,
            window_size=window_size,
            seed=seed,
            hidden_layers=_hidden,
            lr=lr,
            delta=delta if use_drift else 1e9,  # delta enorme ≈ drift nunca dispara
            device=device,
        )

    elif arch == 'soft_reset':
        tiers = COMPOSITIONS.get(composition, COMPOSITIONS['abc'])
        n_tiers = len(tiers)
        loss_f = nn.CrossEntropyLoss()
        ensemble_list = []
        model_configs = []

        for i in range(n_models):
            cfg = tiers[i % n_tiers]
            m = classification.Classifier(
                module=FlexibleNeuralNetwork(n_feat, n_classes, cfg['layers'], cfg['cnn'], None),
                loss_fn=loss_f,
                optimizer_fn=cfg['opt'],
                lr=cfg['lr'],
                device=device,
                is_feature_incremental=False,
            )
            ensemble_list.append(m)
            model_configs.append({'optimizer_fn': cfg['opt'], 'lr': cfg['lr']})

        detector = ADWINChangeDetector(delta=0.001) if use_drift else NoDriftDetector()

        model = ARTESoftResetNN(
            models=ensemble_list,
            model_configs=model_configs,
            drift_detector=detector,
            n_reset_layers=n_reset_layers,
            lambda_val=lambda_val,
            seed=seed,
            window_size=window_size,
        )

    else:
        raise ValueError(f"arch deve ser 'subspace' ou 'soft_reset', recebido: '{arch}'")

    print(f"Arch: {arch} | n_models: {n_models} | Drift: {'ADWIN' if use_drift else 'desativado'}")

    # ------------------------------------------------------------------
    # 5. Métricas + arquivo de saída
    # ------------------------------------------------------------------
    metric_acc     = metrics.Accuracy()
    metric_kappa   = metrics.CohenKappa()
    metric_kappa_m = KappaM()
    metric_gmean   = metrics.GeometricMean()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if arch == 'subspace':
        output_file = (
            f"results/neural/NeuralARTE_{dataset}_subspace"
            f"_n{n_models}_s{seed}_{timestamp}.csv"
        )
        run_id = f"subspace_n{n_models}"
    else:
        drift_tag = 'adwin' if use_drift else 'nodrift'
        output_file = (
            f"results/neural/NeuralARTE_{dataset}_soft_reset"
            f"_{composition}_rl{n_reset_layers}_{drift_tag}_s{seed}_{timestamp}.csv"
        )
        run_id = f"soft_reset_{composition}_rl{n_reset_layers}_{drift_tag}"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Salvando em: {output_file}")

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
    # 6. Loop prequencial
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
        f"Acc: {metric_acc.get():.2%}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runner para ARTESubspaceNN (subspace) e ARTESoftResetNN (soft_reset).'
    )
    parser.add_argument('--arch', type=str, required=True,
                        choices=['subspace', 'soft_reset'],
                        help='Arquitetura: subspace (Random Subspace) ou soft_reset (Reset Parcial)')
    parser.add_argument('--dataset',            type=str, required=True)
    parser.add_argument('--seed',               type=int, default=123456789)
    parser.add_argument('--n_models',           type=int, default=30)
    parser.add_argument('--lambda_val',         type=int, default=6)
    parser.add_argument('--window',             type=int, default=500)
    parser.add_argument('--n_reset_layers',     type=int, default=1,
                        help='[soft_reset only] Camadas lineares finais a reiniciar no drift.')
    parser.add_argument('--composition',        type=str, default='abc',
                        choices=list(COMPOSITIONS.keys()),
                        help='[soft_reset only] Composição do ensemble.')
    parser.add_argument('--lr',                 type=float, default=0.005,
                        help='[subspace only] Learning rate Adam de cada membro.')
    parser.add_argument('--delta',              type=float, default=0.001,
                        help='[subspace only] Sensibilidade do ADWIN (padrão 0.001; menor = menos sensível).')
    parser.add_argument('--datasets_path',      type=str, default=None)
    parser.add_argument('--datasets_path_real', type=str, default=None,
                        help='Caminho fallback (pasta full) quando ARFF não encontrado em datasets_path.')
    parser.add_argument('--device',             type=str, default=None,
                        help='cuda, cuda:0, cuda:1, cpu. Padrão: auto-detect.')
    parser.add_argument('--no_drift',           action='store_true',
                        help='Desativa o detector de drift.')

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        seed=args.seed,
        n_models=args.n_models,
        lambda_val=args.lambda_val,
        window_size=args.window,
        arch=args.arch,
        n_reset_layers=args.n_reset_layers,
        hidden_layers=None,
        lr=args.lr,
        composition=args.composition,
        datasets_path=args.datasets_path,
        datasets_path_fallback=args.datasets_path_real,
        device=args.device,
        delta=args.delta,
        use_drift=not args.no_drift,
    )
