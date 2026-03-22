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
import uuid
from scipy.io import arff
from river import stats, utils, drift, metrics
from river import base, stats, utils, drift, metrics, preprocessing, datasets
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ARTE'))
from metrics import KappaM
from deep_river import classification

# =============================================================================
# CONFIGURAÇÃO DE AMBIENTE — ajuste para local ou servidor remoto
# =============================================================================
DATASETS_PATH = "/home/charan/moa/aldopaim/AdaptiveRandomTreeEnsemble/datasets"  # default local

# =============================================================================
# 1. CARREGAMENTO DE DADOS (Protocolo ARFF Unificado)
# =============================================================================
def get_dataset_universal(dataset_name, seed=42, n_synthetic=None):
    """
    Carregador Universal: L� ARFFs do disco.
    Retorna: X (numpy), y (numpy), n_features, n_classes, nominal_indices
    """
    name = dataset_name.lower()

    files = {
        'airlines':    'airlines.arff',
        'electricity': 'elecNormNew.arff',
        'elec2':       'elecNormNew.arff',
        'covtype':     'covtypeNorm.arff',
        'gassensor':   'gassensor.arff',
        'gmsc':        'GMSC.arff',
        'keystroke':   'keystroke.arff',
        'outdoor':     'outdoor.arff',
        'ozone':       'ozone.arff',
        'rialto':      'rialto.arff',
        'shuttle':     'shuttle.arff',
        'noaa':        'NOAA.arff',
        'agrawal_a':   'agrawal_a.arff',
        'agrawal_g':   'agrawal_g.arff',
        'led_a':       'led_a.arff',
        'led_g':       'led_g.arff',
        'sea_a':       'sea_a.arff',
        'sea_g':       'sea_g.arff',
        'rbf_f':       'rbf_f.arff',
        'rbf_m':       'rbf_m.arff',
    }

    if name not in files:
        raise ValueError(f"Dataset '{name}' desconhecido. Disponíveis: {list(files.keys())}")

    filename = files[name]
    path = os.path.join(DATASETS_PATH, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo {filename} nao encontrado em {DATASETS_PATH}")

    print(f"--- Carregando {filename} ---")
    try:
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')

        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        try:
            y = pd.to_numeric(y)
        except:
            y = pd.Categorical(y).codes

        nominal_attributes = []
        X_final = X.copy()

        for idx, col in enumerate(X.columns):
            if X[col].dtype == object or X[col].dtype.name == 'category':
                nominal_attributes.append(idx)
                X_final[col] = pd.Categorical(X[col]).codes
            else:
                X_final[col] = pd.to_numeric(X_final[col], errors='coerce').fillna(0.0)

        X_np = X_final.values if hasattr(X_final, 'values') else X_final
        y_np = y.values if hasattr(y, 'values') else y

        # Garante labels 0-indexados contíguos (CrossEntropyLoss exige [0, n_classes-1])
        unique_labels = np.unique(y_np)
        if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
            label_map = {v: i for i, v in enumerate(unique_labels)}
            y_np = np.array([label_map[v] for v in y_np])

        return X_np, y_np, X_np.shape[1], len(np.unique(y_np)), nominal_attributes

    except Exception as e:
        print(f"[ERRO FATAL] lendo {filename}: {e}")
        raise

def apply_one_hot_encoding(X, nominal_indices):
    """
    Aplica OHE apenas nas colunas nominais indicadas.
    Essencial para Redes Neurais performarem bem em datasets como Airlines.
    """
    if not nominal_indices:
        return X

    print(f"Aplicando One-Hot Encoding em {len(nominal_indices)} colunas...")
    df = pd.DataFrame(X)
    
    # OHE nas colunas especificadas
    # drop_first=True evita colinearidade perfeita (opcional)
    df = pd.get_dummies(df, columns=nominal_indices, dtype=float)
    
    X_new = df.values.astype(np.float32)
    print(f"   -> Expansao de Features: {X.shape[1]} -> {X_new.shape[1]}")
    return X_new

# =============================================================================
# 2. UTILIT�RIOS (Log e Scaler)
# =============================================================================
def log_results_to_csv(filename, data_dict):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)

class FastIncrementalScaler:
    """Scaler Incremental Welford (Vetorizado)."""
    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float32)
        self.M2 = np.zeros(n_features, dtype=np.float32)

    def learn_one(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform_one(self, x):
        if self.n < 2: return x
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var) + 1e-8
        return (x - self.mean) / std

# =============================================================================
# 3. REDE NEURAL E ENSEMBLE (Sua Implementa��o Otimizada)
# =============================================================================
class FlexibleNeuralNetwork(nn.Module):
    def __init__(self, n_features, n_classes, hidden_layers=[32], use_cnn=False, projection_matrix=None):
        super().__init__()
        self.n_features = n_features
        self.use_cnn = use_cnn
        if projection_matrix is not None:
            self.register_buffer('projection', projection_matrix.to(torch.float32))
        else:
            self.projection = None
            
        # Define dimens�o de entrada (com ou sem proje��o, a dimens�o � mantida ou alterada externamente)
        # Se houver proje��o, assume-se que ela projeta de n_features -> n_features (rota��o)
        self.in_dim = (8 * n_features) if use_cnn else n_features
        
        if use_cnn:
            self.cnn_block = nn.Sequential(nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(), nn.Flatten())
        else:
            self.cnn_block = None
            
        layers = []
        curr = self.in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.ReLU())
            curr = h
        layers.append(nn.Linear(curr, n_classes))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        x = x.to(torch.float32)
        
        if self.projection is not None: 
            x = torch.matmul(x, self.projection)
            
        if self.use_cnn: 
            x = self.cnn_block(x.unsqueeze(1))
            
        return self.mlp_head(x)

class ARTELight(base.Ensemble, base.Classifier):
    def __init__(self, models, model_types, drift_detector, lambda_val=6, seed=42, window_size=500):
        super().__init__(models=models)
        self.lambda_val = lambda_val
        self._rng = np.random.RandomState(seed)
        self.drift_detector = drift_detector
        
        # Clones independentes
        self._detectors = [drift_detector.clone() for _ in range(len(models))]
        self._acc_windows = [utils.Rolling(stats.Mean(), window_size=window_size) for _ in range(len(models))]
        self._total_drifts = 0
        
        # Instrumentação
        self.model_types = model_types
        self.stats_correct = {t: 0 for t in set(model_types)}
        self.stats_drifts = {t: 0 for t in set(model_types)}

    def learn_one(self, x, y):
        # x: Tensor na GPU [feat] ou [1, feat]
        # y: Inteiro
        
        # Garante dimensão de batch [1, feat] para o PyTorch
        if isinstance(x, torch.Tensor) and x.ndim == 1:
            x_in = x.unsqueeze(0)
        else:
            x_in = x

        any_drift = False

        for i, model in enumerate(self.models):
            # 1. Predição para monitoramento (Test-then-Train)
            # Bypass Deep River (que espera dict) — chama module diretamente como em predict_proba_one
            model.module.eval()
            with torch.inference_mode():
                logits = model.module(x_in)
                y_pred = torch.argmax(logits, dim=1).item()
            
            correct = (y == y_pred)
            
            # Atualiza stats e detectors
            if correct: self.stats_correct[self.model_types[i]] += 1
            self._detectors[i].update(0 if correct else 1)
            self._acc_windows[i].update(1 if correct else 0)
            
            # 2. Treino (Boosting via Poisson)
            if not correct:
                k = self._rng.poisson(self.lambda_val)
                if k > 0:
                    # OTIMIZAÇÃO: Repeat na GPU ao invés de DataFrame Pandas
                    x_boost = x_in.repeat(k, 1)
                    y_boost = torch.tensor([y] * k, device=x_in.device, dtype=torch.long)
                    # model.learn_many(x_boost, y_boost)
                    # --- CORREÇÃO DE TREINO (Bypass Deep River) ---
                    model = self.models[i]
                    model.module.train() # Modo treino
                    
                    # Zera gradientes
                    model.optimizer.zero_grad()
                    
                    # Forward
                    y_pred_logits = model.module(x_boost)
                    
                    # Loss Calculation
                    loss = model.loss_fn(y_pred_logits, y_boost)
                    
                    # Backward & Step
                    loss.backward()
                    model.optimizer.step()
                    
                    

            # 3. Drift Reset
            if self._detectors[i].drift_detected:
                self._total_drifts += 1
                self.stats_drifts[self.model_types[i]] += 1
                self.models[i] = model.clone() 
                self._detectors[i] = self.drift_detector.clone()
                self._acc_windows[i] = utils.Rolling(stats.Mean(), window_size=100)
                any_drift = True
                
        return any_drift

    def learn_many(self, X, y):
        """
        Treinamento em Lote Otimizado (GPU).
        X: Tensor [Batch, Features]
        y: Tensor [Batch] (Long)
        """
        batch_size = X.shape[0]
        device = X.device
        
        # Garante que y é LongTensor para a Loss function
        y = y.long()

        for i, model in enumerate(self.models):
            
            # --- 1. Predição do Lote (Bypass Deep River para velocidade) ---
            model.module.eval()
            with torch.no_grad():
                logits = model.module(X)
                # Pega a classe predita (argmax)
                y_pred = torch.argmax(logits, dim=1)
            
            # --- 2. Máscara de Erros (Quem errou?) ---
            # Retorna vetor booleano [True, False, True...]
            incorrect_mask = (y_pred != y)
            
            # --- 3. Atualização de Drift e Stats (Gargalo CPU necessário) ---
            # Infelizmente o ADWIN do River é sequencial e roda na CPU.
            # Convertemos para numpy apenas os bits necessários para atualizar os detectores.
            if incorrect_mask.any():
                incorrect_cpu = incorrect_mask.cpu().numpy()
                
                # Loop rápido apenas para atualizar estatísticas
                for is_incorrect in incorrect_cpu:
                    val = 1 if is_incorrect else 0
                    self._detectors[i].update(val)
                    self._acc_windows[i].update(0 if is_incorrect else 1)
                    
                # Contabiliza acertos no placar global
                # (Total do batch - Total de erros)
                n_errors = incorrect_cpu.sum()
                self.stats_correct[self.model_types[i]] += (batch_size - n_errors)

            else:
                # Se acertou tudo, atualiza detectors com 0
                for _ in range(batch_size):
                    self._detectors[i].update(0)
                    self._acc_windows[i].update(1)
                self.stats_correct[self.model_types[i]] += batch_size

            # --- 4. Lógica de Drift (Reset) ---
            if self._detectors[i].drift_detected:
                self._total_drifts += 1
                self.stats_drifts[self.model_types[i]] += 1
                
                # Reset do modelo (Clone limpo)
                self.models[i] = model.clone() 
                self._detectors[i] = self.drift_detector.clone()
                self._acc_windows[i] = utils.Rolling(stats.Mean(), window_size=100)
                
                # Se houve drift, forçamos um treino com peso 1 em todo o batch
                # para o novo modelo já nascer vendo dados
                k_vector = torch.ones(batch_size, device=device, dtype=torch.long)
                
            else:
                # --- 5. Lógica de Boosting (Vectorized Poisson) ---
                # Queremos treinar apenas onde o modelo ERROU (conforme sua lógica original)
                # k ~ Poisson(lambda) para cada instância
                
                # Gera Poisson para o lote todo
                lambda_tensor = torch.full((batch_size,), self.lambda_val, device=device, dtype=torch.float)
                k_raw = torch.poisson(lambda_tensor)
                
                # Aplica a máscara: Zera o k se o modelo acertou a instância
                # k = Poisson se errou, 0 se acertou
                k_vector = (k_raw * incorrect_mask.float()).long()

            # --- 6. Treino Efetivo (Backpropagation) ---
            # Se a soma de k for 0, ninguém precisa de treino nesse lote -> Pula
            if k_vector.sum() > 0:
                
                # OVERSAMPLING NA GPU (Repetição eficiente)
                # Se k=[0, 2, 1], repetimos a instância 2 duas vezes e a 3 uma vez.
                X_train = torch.repeat_interleave(X, k_vector, dim=0)
                y_train = torch.repeat_interleave(y, k_vector, dim=0)
                
                # Setup do Treino
                model.module.train()
                model.optimizer.zero_grad()
                
                # Forward Pass
                logits_train = model.module(X_train)
                loss = model.loss_fn(logits_train, y_train)
                
                # Backward Pass
                loss.backward()
                model.optimizer.step()
                
    @torch.inference_mode()
    def predict_proba_one(self, x):
        # x: Tensor na GPU [1, n_feat]
        if isinstance(x, torch.Tensor) and x.ndim == 1:
            x = x.unsqueeze(0)

        # Lógica de Votação Dinâmica
        accs = [w.get() for w in self._acc_windows]
        avg = sum(accs)/len(accs) if accs else 0
        idx = [i for i, a in enumerate(accs) if a >= avg] or range(len(self.models))
        
        votes = collections.Counter()
        
        for i in idx:
            model = self.models[i]
            
            # --- CORREÇÃO DO ERRO ---
            # Bypass no wrapper do Deep River. Chamamos a rede (module) diretamente.
            # O wrapper deep_river.Classifier guarda a rede em self.module
            
            # Garante modo de avaliação
            model.module.eval()
            
            # Inferência direta (Rápida!)
            # A rede retorna logits (raw scores). Precisamos aplicar Softmax.
            logits = model.module(x) 
            proba_tensor = torch.softmax(logits, dim=1)
            
            # Extrai probabilidades para CPU (necessário para o Counter do Python)
            # Como é binary/multiclass, pegamos a lista de probabilidades
            probas = proba_tensor.cpu().numpy()[0]
            
            # Mapeia índice -> probabilidade (0: p0, 1: p1...)
            # Assume classes 0, 1, 2... ordenadas
            for class_idx, prob_val in enumerate(probas):
                votes[class_idx] += prob_val / len(idx)
                
        return votes

    @torch.inference_mode()
    def predict_one(self, x):
        """
        Retorna a classe final (Hard Label) baseada na agregação de probabilidades.
        O main não precisa saber como isso é calculado.
        """
        y_proba = self.predict_proba_one(x)
        if y_proba:
            return max(y_proba, key=y_proba.get)
        return 0 # Fallback
        
    @property
    def total_drifts(self): 
        return self._total_drifts

# =============================================================================
# 4. EXECU��O
# =============================================================================
class NoDriftDetector:
    """Detector nulo — nunca dispara drift. Permite medir adaptação natural das redes."""
    def update(self, x): pass
    def clone(self): return NoDriftDetector()
    @property
    def drift_detected(self): return False


# Presets de composição do ensemble
COMPOSITIONS = {
    "current": [
        # MLP_Simple + MLP_CNN + MLP_Proj (configuração atual)
        {"type": "MLP_Simple", "opt": optim.SGD,  "lr": 0.05,  "layers": [128, 64], "cnn": False, "proj": False},
        {"type": "MLP_CNN",    "opt": optim.Adam, "lr": 0.01,  "layers": [64],       "cnn": True,  "proj": False},
        {"type": "MLP_Proj",   "opt": optim.Adam, "lr": 0.005, "layers": [256, 128], "cnn": False, "proj": True},
    ],
    "abc": [
        # A: "Veloz"      — SGD, LR alto, raso   → reage rápido a drifts abruptos
        {"type": "MLP_Fast",  "opt": optim.SGD,  "lr": 0.05,  "layers": [64],        "cnn": False, "proj": False},
        # B: "Analítico"  — Adam, LR baixo, profundo → aprende fronteiras complexas
        {"type": "MLP_Deep",  "opt": optim.Adam, "lr": 0.001, "layers": [256, 128, 64], "cnn": False, "proj": False},
        # C: "Equilibrado" — Adam, LR médio, médio → robusto geral
        {"type": "MLP_Mid",   "opt": optim.Adam, "lr": 0.01,  "layers": [128, 64],   "cnn": False, "proj": False},
    ],
}


def main_neural_arte(dataset, seed, n_models, lambda_val, window_size, datasets_path=None, device=None, batch_size=32, use_projection=True, composition="current", use_drift=True):
    
    global DATASETS_PATH
    if datasets_path:
        DATASETS_PATH = datasets_path

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Iniciando Neural ARTE (device: {device}) ---")
    
    # 1. Carrega Dados (ARFF)
    try:
        X_all, y_all, n_feat_raw, n_classes, nom_indices = get_dataset_universal(dataset, seed=seed)
    except Exception as e:
        print(f"Erro carregando dataset: {e}")
        return

    # 2. Pr�-processamento OHE (Fundamental para MLPs)
    if nom_indices:
        X_all = apply_one_hot_encoding(X_all, nom_indices)
    
    n_feat = X_all.shape[1]
    print(f"Dataset: {dataset} | Inst: {len(X_all)} | Feat: {n_feat} | Classes: {n_classes}")

    # 3. Configuração do Ensemble
    tiers = COMPOSITIONS.get(composition, COMPOSITIONS["current"])
    n_tiers = len(tiers)
    loss_f = nn.CrossEntropyLoss()
    ensemble_list = []
    model_types_list = []
    torch.manual_seed(seed)

    print(f"Composição: {composition} | Drift detector: {'ADWIN' if use_drift else 'desativado'}")

    for i in range(n_models):
        cfg = tiers[i % n_tiers]
        proj = torch.randn(n_feat, n_feat) if (cfg["proj"] and use_projection) else None
        if proj is not None:
            proj, _ = torch.linalg.qr(proj)

        m = classification.Classifier(
            module=FlexibleNeuralNetwork(n_feat, n_classes, cfg["layers"], cfg["cnn"], proj),
            loss_fn=loss_f,
            optimizer_fn=cfg["opt"],
            lr=cfg["lr"],
            device=device,
            is_feature_incremental=False
        )
        ensemble_list.append(m)
        model_types_list.append(cfg["type"])

    detector = drift.ADWIN(delta=0.001) if use_drift else NoDriftDetector()
    model = ARTELight(
        models=ensemble_list,
        model_types=model_types_list,
        drift_detector=detector,
        lambda_val=lambda_val,
        seed=seed,
        window_size=window_size
    )

    # 4. M�tricas e Log
    metric_acc = metrics.Accuracy()
    metric_kappa = metrics.CohenKappa()
    metric_kappa_m = KappaM()
    metric_gmean = metrics.GeometricMean()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    drift_tag = "nodrift" if not use_drift else "adwin"
    output_file = f"results/neural/NeuralARTE_{dataset}_{composition}_{drift_tag}_s{seed}_{timestamp}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Salvando em: {output_file}")
    
    latencies = []
    start_total = time.time()

    def save_snapshot(current_count, force=False):
        ram = psutil.Process().memory_info().rss / (1024 * 1024)
        vram = torch.cuda.memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0

        avg_lat = 0.0
        if latencies:
            slice_size = min(len(latencies), 2000)
            avg_lat = sum(latencies[-slice_size:]) / slice_size

        stats_dict = {
            "Run_ID": f"{composition}_{drift_tag}",
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "Instancia": current_count,
            "Dataset": dataset,
            "Accuracy": metric_acc.get(),
            "Kappa": metric_kappa.get(),
            "KappaM": metric_kappa_m.get(),
            "GMean": metric_gmean.get(),
            "Latencia_ms": avg_lat,
            "Drifts": model.total_drifts,
            "RAM_MB": ram,
            "VRAM_MB": vram
        }
        log_results_to_csv(output_file, stats_dict)
        if force or current_count % 10000 == 0:
            print(f"[{dataset}] Inst: {current_count} | Acc: {metric_acc.get():.2%} | Kappa: {metric_kappa.get():.2f} | Drifts: {model.total_drifts} | RAM: {ram:.0f}MB")

    # --- PRÉ-ESCALONAMENTO E PRÉ-TENSORIZAÇÃO (elimina round-trips CPU→GPU por instância) ---
    print("Pré-escalonando dataset...")
    X_scaled_all = np.zeros_like(X_all, dtype=np.float32)
    scaler_pre = FastIncrementalScaler(n_feat)
    for i in range(len(X_all)):
        scaler_pre.learn_one(X_all[i])
        X_scaled_all[i] = scaler_pre.transform_one(X_all[i])
    X_gpu = torch.tensor(X_scaled_all, device=device, dtype=torch.float32)
    print("Pronto. Iniciando loop prequencial...")

    # --- LOOP PRINCIPAL (Test-then-Train por instância, igual ao notebook) ---
    log_interval = 2000

    for count in range(len(X_all)):
        y = int(y_all[count])

        t0 = time.perf_counter()

        # 1. Tensor já na GPU (sem transferência por instância)
        x_tensor = X_gpu[count]

        # 2. Predict
        y_pred = model.predict_one(x_tensor)

        t_pred = time.perf_counter() - t0

        # 3. Update Metrics
        metric_acc.update(y, y_pred)
        metric_kappa.update(y, y_pred)
        metric_kappa_m.update(y, y_pred)
        metric_gmean.update(y, y_pred)

        # 4. Learn
        t1 = time.perf_counter()
        model.learn_one(x_tensor, y)
        t_learn = time.perf_counter() - t1

        latencies.append((t_pred + t_learn) * 1000)

        # 5. Log
        if (count + 1) % log_interval == 0:
            save_snapshot(count + 1)

    # Log Final Forçado
    if len(X_all) % log_interval != 0:
        save_snapshot(len(X_all), force=True)

    print(f" Fim {dataset}. Tempo Total: {(time.time() - start_total):.1f}s | Acc: {metric_acc.get():.2%}")

if __name__ == "__main__":
    # Exemplo de uso via CLI ou chamada direta
    # python3 main_neural_arte.py --dataset agrawal_a
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       type=str,   default='electricity')
    parser.add_argument('--seed',          type=int,   default=123456789)
    parser.add_argument('--n_models',      type=int,   default=30)
    parser.add_argument('--lambda_val',    type=int,   default=6)
    parser.add_argument('--window',        type=int,   default=500)
    parser.add_argument('--datasets_path', type=str,   default=None,
                        help='Caminho para a pasta com os ARFFs. Sobrescreve o default do código.')
    parser.add_argument('--device',        type=str,   default=None,
                        help='Device PyTorch: cuda, cuda:0, cuda:1, cpu. Default: auto-detect.')
    parser.add_argument('--no_projection', action='store_true',
                        help='Desativa projeção ortogonal no tier MLP_Proj.')
    parser.add_argument('--composition',   type=str,   default='current',
                        choices=list(COMPOSITIONS.keys()),
                        help='Composição do ensemble: current (padrão) | abc (Veloz+Analítico+Equilibrado).')
    parser.add_argument('--no_drift',      action='store_true',
                        help='Desativa detector de drift (mede adaptação natural das redes).')
    args = parser.parse_args()

    main_neural_arte(args.dataset, args.seed, args.n_models, args.lambda_val, args.window,
                     datasets_path=args.datasets_path, device=args.device,
                     use_projection=not args.no_projection,
                     composition=args.composition,
                     use_drift=not args.no_drift)