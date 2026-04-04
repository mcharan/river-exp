"""
Comparação ADWIN min_window_length=5 vs =10 nos experimentos ARTE.

Lê todos os CSVs de results/arte/ com padrão ARTE_CPU_{dataset}_mw{5|10}_*
e exibe tabelas comparativas com referência MOA quando disponível.

Uso:
    python analysis/compare_mw.py
    python analysis/compare_mw.py --folder results/arte --metric acc
    python analysis/compare_mw.py --full
"""

import os
import re
import glob
import argparse
import pandas as pd

# =============================================================================
# REFERÊNCIA MOA (Java) — Tabela 14 do artigo / nota_metodologica_adwin.md
# Valores de acurácia (%) com seed=123456789, n_models=100, delta=1e-3
# =============================================================================
MOA_REF = {
    # Fonte: nota_metodologica_adwin.md
    # seed=123456789, n_models=100, delta=1e-3
    'rbf_m':     87.71,
    'rbf_f':     79.86,
    'agrawal_a': 79.50,
    'agrawal_g': 75.49,
}


# =============================================================================
# PARSING DO NOME DO ARQUIVO
# ARTE_CPU_{dataset}_mw{mw}_s{seed}_{timestamp}.csv
# =============================================================================
def parse_filename(filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]
    m = re.match(r'ARTE_CPU_(.+)_mw(\d+)_s(\d+)_(\d{8}_\d{6})$', base)
    if not m:
        return None
    return {
        'dataset': m.group(1),
        'mw':      int(m.group(2)),
        'seed':    int(m.group(3)),
    }


# =============================================================================
# LEITURA DOS CSVs
# =============================================================================
def load_results(folder):
    files = glob.glob(os.path.join(folder, 'ARTE_CPU_*.csv'))
    rows = []
    for f in sorted(files):
        meta = parse_filename(f)
        if meta is None:
            continue
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            last = df.iloc[-1]

            acc     = last.get('Accuracy',   float('nan')) * 100
            kappa   = last.get('Kappa',      float('nan'))
            kappa_m = last.get('Kappa_M',    float('nan'))
            gmean   = last.get('GMean',      float('nan')) * 100
            drifts  = last.get('Drifts',     float('nan'))
            lat     = df['Latencia_ms'].mean()
            ram     = df['RAM_MB'].max()
            n_inst  = int(last.get('Instancia', 0))

            total_time = 0.0
            if 'Time' in df.columns and len(df) > 1:
                try:
                    t0 = pd.to_datetime(df.iloc[0]['Time'],  format='%H:%M:%S')
                    t1 = pd.to_datetime(last['Time'],        format='%H:%M:%S')
                    if t1 < t0:
                        t1 += pd.Timedelta(days=1)
                    total_time = (t1 - t0).total_seconds() / 60
                except Exception:
                    pass

            rows.append({
                'dataset':    meta['dataset'],
                'mw':         meta['mw'],
                'seed':       meta['seed'],
                'n_inst':     n_inst,
                'acc':        acc,
                'kappa':      kappa,
                'kappa_m':    kappa_m,
                'gmean':      gmean,
                'drifts':     int(drifts) if not pd.isna(drifts) else None,
                'lat_ms':     lat,
                'ram_mb':     ram,
                'time_min':   total_time,
                'file':       os.path.basename(f),
            })
        except Exception as e:
            print(f'[ERRO] {os.path.basename(f)}: {e}')

    return pd.DataFrame(rows)


# =============================================================================
# TABELAS
# =============================================================================
def tabela_principal(df, metric='acc'):
    metric_label = {
        'acc':    'Acurácia (%)',
        'kappa':  'Kappa',
        'drifts': 'Drifts',
        'gmean':  'GMean (%)',
        'lat_ms': 'Lat (ms)',
    }
    label = metric_label.get(metric, metric)

    print(f"\n{'='*75}")
    print(f" COMPARAÇÃO mw=5 vs mw=10  —  {label}")
    print(f"{'='*75}")
    print(f"{'Dataset':<14} {'mw=5':>8} {'mw=10':>8} {'Δ(10−5)':>9}", end='')
    if metric == 'acc':
        print(f" {'MOA ref':>8} {'Δ(5−MOA)':>9} {'Δ(10−MOA)':>10}", end='')
    print()
    print('-' * 75)

    datasets = sorted(df['dataset'].unique())
    for ds in datasets:
        sub = df[df['dataset'] == ds]
        v5  = sub[sub['mw'] == 5][metric].values
        v10 = sub[sub['mw'] == 10][metric].values

        s5  = f'{v5[0]:8.3f}'  if len(v5)  else f'{"—":>8}'
        s10 = f'{v10[0]:8.3f}' if len(v10) else f'{"—":>8}'

        if len(v5) and len(v10):
            delta = v10[0] - v5[0]
            sign = '+' if delta >= 0 else ''
            sdelta = f'{sign}{delta:8.3f}'
        else:
            sdelta = f'{"—":>9}'

        print(f'{ds:<14} {s5} {s10} {sdelta}', end='')

        if metric == 'acc' and ds in MOA_REF:
            moa = MOA_REF[ds]
            d5  = (v5[0]  - moa) if len(v5)  else float('nan')
            d10 = (v10[0] - moa) if len(v10) else float('nan')
            sign5  = '+' if d5  >= 0 else ''
            sign10 = '+' if d10 >= 0 else ''
            print(f' {moa:8.2f} {sign5}{d5:8.3f} {sign10}{d10:9.3f}', end='')
        elif metric == 'acc':
            print(f' {"—":>8} {"—":>9} {"—":>10}', end='')

        print()


def tabela_drifts(df):
    print(f"\n{'='*55}")
    print(f" DRIFTS DETECTADOS: mw=5 vs mw=10")
    print(f"{'='*55}")
    print(f"{'Dataset':<14} {'mw=5':>8} {'mw=10':>8} {'Redução%':>10}")
    print('-' * 55)

    for ds in sorted(df['dataset'].unique()):
        sub = df[df['dataset'] == ds]
        d5  = sub[sub['mw'] == 5]['drifts'].values
        d10 = sub[sub['mw'] == 10]['drifts'].values

        s5  = f'{d5[0]:8d}'  if len(d5)  and d5[0]  is not None else f'{"—":>8}'
        s10 = f'{d10[0]:8d}' if len(d10) and d10[0] is not None else f'{"—":>8}'

        if len(d5) and len(d10) and d5[0] and d5[0] > 0:
            red = (d5[0] - d10[0]) / d5[0] * 100
            sign = '+' if red >= 0 else ''
            sred = f'{sign}{red:9.1f}%'
        else:
            sred = f'{"—":>10}'

        print(f'{ds:<14} {s5} {s10} {sred}')


def tabela_completa(df):
    print(f"\n{'='*100}")
    print(f" TABELA COMPLETA")
    print(f"{'='*100}")
    cols = ['dataset', 'mw', 'n_inst', 'acc', 'kappa', 'kappa_m', 'gmean',
            'drifts', 'lat_ms', 'ram_mb', 'time_min']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values(['dataset', 'mw']).round(3).to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparação ADWIN mw=5 vs mw=10')
    parser.add_argument('--folder', default='results/arte',
                        help='Pasta com os CSVs (padrão: results/arte)')
    parser.add_argument('--metric', default='acc',
                        choices=['acc', 'kappa', 'drifts', 'gmean', 'lat_ms'],
                        help='Métrica principal (padrão: acc)')
    parser.add_argument('--full', action='store_true',
                        help='Exibe tabela completa linha a linha')
    args = parser.parse_args()

    df = load_results(args.folder)
    if df.empty:
        print(f"Nenhum resultado encontrado em '{args.folder}'.")
        exit(0)

    print(f"\nArquivos carregados : {len(df)}")
    print(f"Datasets            : {sorted(df['dataset'].unique())}")
    print(f"Configurações mw    : {sorted(df['mw'].unique())}")
    print(f"Instâncias (amostra): {df.groupby('dataset')['n_inst'].max().to_dict()}")

    tabela_principal(df, metric=args.metric)
    tabela_drifts(df)

    if args.full:
        tabela_completa(df)
