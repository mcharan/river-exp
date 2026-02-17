import numpy as np
import pandas as pd
import os
from scipy.io import arff
from river import datasets

# =============================================================================
# 3. UTILS DE LOGGING
# =============================================================================
def log_results_to_csv(filename, stats_dict):
    df = pd.DataFrame([stats_dict])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


def get_dataset_universal(dataset_name, seed=42, n_synthetic=None):
    """
    Carregador Universal: Le ARFFs do disco (Reais e Sinteticos do MOA).
    Retorna: X (numpy), y (numpy), n_features, n_classes, nominal_indices
    """
    name = dataset_name.lower()
    paim_path = "/home/charan/moa/aldopaim/AdaptiveRandomTreeEnsemble/datasets" 
    
    # Mapeamento Completo
    files = {
        # --- Datasets Reais ---
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
        
        # --- Sinteticos (Gerados pelo MOA CLI) ---
        'agrawal_a':   'agrawal_a.arff',
        'agrawal_g':   'agrawal_g.arff',
        'led_a':       'led_a.arff',
        'led_g':       'led_g.arff',
        'sea_a':       'sea_a.arff',
        'sea_g':       'sea_g.arff',
        'rbf_f':       'rbf_f.arff',     
        'rbf_m':       'rbf_m.arff',     
        'mixed_a':     'mixed.arff'      
    }

    if name in files:
        filename = files[name]
        path = os.path.join(paim_path, filename)
        
        if not os.path.exists(path):
            if name == 'electricity':
                print(f"[AVISO]: {filename} nao encontrado. Usando River.datasets.Elec2().")
                X, y, nf, nc = _load_river_dataset(datasets.Elec2())
                return X, y, nf, nc, []
            raise FileNotFoundError(f"Arquivo {filename} nao encontrado em {paim_path}")

        print(f"--- Carregando {filename} ---")
        try:
            data, meta = arff.loadarff(path)
            df = pd.DataFrame(data)
            
            # 1. Decodifica bytes
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.decode('utf-8')

            # 2. Separa X e y
            target_col = df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # 3. Tratamento do Target (Blindado)
            try:
                y = pd.to_numeric(y)
            except:
                y = pd.Categorical(y).codes 
            # Se virou codes, ja eh numpy. Se eh numeric, eh Series.

            # 4. Tratamento das Features (X)
            nominal_attributes = []
            X_final = X.copy()

            for idx, col in enumerate(X.columns):
                if X[col].dtype == object or X[col].dtype.name == 'category':
                    nominal_attributes.append(idx)
                    X_final[col] = pd.Categorical(X[col]).codes
                else:
                    X_final[col] = pd.to_numeric(X_final[col], errors='coerce').fillna(0.0)

            print(f"   >> {len(X_final)} instancias carregadas. Nominais detectados: {len(nominal_attributes)}")
            
            # --- CORRECAO SEGURA NO RETORNO ---
            # Garante X como numpy
            X_np = X_final.values if hasattr(X_final, 'values') else X_final
            
            # Garante y como numpy (trata caso Series vs Array)
            y_np = y.values if hasattr(y, 'values') else y
            
            return X_np, y_np, X_np.shape[1], len(np.unique(y_np)), nominal_attributes

        except Exception as e:
            print(f"[ERRO FATAL] lendo {filename}: {e}")
            raise

    # Fallbacks River
    if name == 'covtype_river': 
        X, y, nf, nc = _load_river_dataset(datasets.Covertype())
        return X, y, nf, nc, []

    raise ValueError(f"Dataset '{name}' desconhecido ou arquivo nao mapeado.")