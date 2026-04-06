"""
results_soft_reset.py — Tabela de resultados do ARTESoftResetNN.

Lê os CSVs de results/neural/ com padrão:
    NeuralARTE_{dataset}_soft_reset_{composition}_rl{n}_adwin_s{seed}_{ts}.csv

Uso:
    python analysis/results_soft_reset.py
    python analysis/results_soft_reset.py --folder results/neural --composition abc
"""

import pandas as pd
import glob
import os
import re
import argparse


# =============================================================================
# PARSING
# NeuralARTE_{dataset}_soft_reset_{composition}_rl{n}_{drift}_s{seed}_{ts}.csv
# =============================================================================
def parse_filename(filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]
    m = re.match(
        r'NeuralARTE_(.+)_soft_reset_([^_]+(?:_[^_]+)*)_rl(\d+)_(adwin|nodrift)_s(\d+)_(\d{8}_\d{6})$',
        base
    )
    if not m:
        return None

    # composição pode conter '_' (ex: abc_proj) — capturamos tudo entre soft_reset_ e _rl
    # O regex acima é guloso; refinamos separando pelo _rl\d+
    m2 = re.match(
        r'NeuralARTE_(.+)_soft_reset_(.+)_rl(\d+)_(adwin|nodrift)_s(\d+)_(\d{8}_\d{6})$',
        base
    )
    if not m2:
        return None

    return {
        'dataset':      m2.group(1),
        'composition':  m2.group(2),
        'n_reset':      int(m2.group(3)),
        'drift':        m2.group(4),
        'seed':         int(m2.group(5)),
        'timestamp':    m2.group(6),
    }


# =============================================================================
# LEITURA
# =============================================================================
def load_results(folder, composition=None, drift=None, seed=None):
    files = glob.glob(os.path.join(folder, "NeuralARTE_*_soft_reset_*.csv"))
    rows = []
    for f in sorted(files):
        meta = parse_filename(f)
        if meta is None:
            continue
        if composition and meta['composition'] != composition:
            continue
        if drift and meta['drift'] != drift:
            continue
        if seed and meta['seed'] != seed:
            continue
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            last = df.iloc[-1]

            acc    = float(last.get('Accuracy', float('nan'))) * 100
            kappa  = float(last.get('Kappa',    float('nan')))
            kappam = float(last.get('Kappa_M', last.get('KappaM', float('nan'))))
            drifts = last.get('Drifts', float('nan'))
            ram    = float(last.get('RAM_MB',   float('nan')))
            lat    = df['Latencia_ms'].mean() if 'Latencia_ms' in df.columns else float('nan')

            total_time = float('nan')
            if 'Time' in df.columns:
                try:
                    t0 = pd.to_datetime(df.iloc[0]['Time'], format='%H:%M:%S')
                    t1 = pd.to_datetime(last['Time'],       format='%H:%M:%S')
                    if t1 < t0:
                        t1 += pd.Timedelta(days=1)
                    total_time = (t1 - t0).total_seconds() / 60
                except Exception:
                    pass

            rows.append({
                'dataset':     meta['dataset'],
                'composition': meta['composition'],
                'n_reset':     meta['n_reset'],
                'drift':       meta['drift'],
                'seed':        meta['seed'],
                'timestamp':   meta['timestamp'],
                'acc':         acc,
                'kappa':       kappa,
                'kappam':      kappam,
                'drifts':      int(drifts) if not pd.isna(drifts) else None,
                'ram_mb':      ram,
                'lat_ms':      lat,
                'time_min':    total_time,
                'file':        os.path.basename(f),
            })
        except Exception as e:
            print(f"[ERRO] {os.path.basename(f)}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Múltiplas rodadas → mantém a mais recente por (dataset, composition, n_reset, drift)
    df = (df.sort_values('timestamp')
            .groupby(['dataset', 'composition', 'n_reset', 'drift'])
            .last()
            .reset_index())
    return df


# =============================================================================
# TABELAS
# =============================================================================
def tabela_principal(df):
    print(f"\n{'='*100}")
    print(f" RESULTADOS — ARTESoftResetNN")
    print(f"{'='*100}")

    t = df.sort_values(['composition', 'n_reset', 'drift', 'dataset'])

    header = (
        f"{'DATASET':<15} | {'COMP':<10} | {'RL':>2} | {'DRIFT':<8} | "
        f"{'Acc':>8} | {'Kappa':>6} | {'KappaM':>7} | "
        f"{'Drifts':>6} | {'RAM(MB)':>8} | {'Lat(ms)':>8} | {'t(min)':>7}"
    )
    print(header)
    print("-" * len(header))

    for _, r in t.iterrows():
        drifts_s = str(r['drifts']) if r['drifts'] is not None else "N/A"
        print(
            f"{r['dataset']:<15} | {r['composition']:<10} | {r['n_reset']:>2} | {r['drift']:<8} | "
            f"{r['acc']:>7.2f}% | {r['kappa']:>6.3f} | {r['kappam']:>7.3f} | "
            f"{drifts_s:>6} | {r['ram_mb']:>8.1f} | {r['lat_ms']:>8.3f} | {r['time_min']:>7.1f}"
        )


def tabela_pivot(df, metric='acc'):
    """Pivot: linhas=dataset, colunas=(composition, n_reset). Só drift=adwin."""
    sub = df[df['drift'] == 'adwin'].copy()
    if sub.empty:
        print("\nNenhum resultado com drift=adwin.")
        return

    sub['config'] = sub['composition'] + '_rl' + sub['n_reset'].astype(str)
    pivot = sub.pivot_table(index='dataset', columns='config', values=metric, aggfunc='mean')

    label = {'acc': 'Acurácia (%)', 'kappa': 'Kappa', 'drifts': 'Drifts', 'lat_ms': 'Lat (ms)'}
    print(f"\n{'='*70}")
    print(f" PIVOT — {label.get(metric, metric)} (drift=adwin)")
    print(f"{'='*70}")
    print(pivot.round(3).to_string())

    # Resumo: média por config
    print(f"\n  Médias por configuração:")
    for col in pivot.columns:
        print(f"    {col:<20}: {pivot[col].mean():.3f}")


def tabela_drift_vs_nodrift(df):
    """Compara adwin vs nodrift para cada (dataset, composition, n_reset)."""
    pivot = df.pivot_table(
        index=['dataset', 'composition', 'n_reset'],
        columns='drift', values='acc', aggfunc='mean'
    )
    if 'adwin' not in pivot.columns or 'nodrift' not in pivot.columns:
        print("\n  Faltam resultados de uma das variantes (adwin ou nodrift).")
        return

    pivot['Δ (nodrift−adwin)'] = pivot['nodrift'] - pivot['adwin']
    print(f"\n{'='*70}")
    print(f" DRIFT DETECTOR: adwin vs nodrift — Acurácia (%)")
    print(f"{'='*70}")
    print(pivot.round(3).to_string())
    print(f"\n  Média Δ: {pivot['Δ (nodrift−adwin)'].mean():+.3f}pp")
    print(f"  Positivo = sem drift melhor | Negativo = com drift melhor")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resultados ARTESoftResetNN')
    parser.add_argument('--folder',      default='results/neural')
    parser.add_argument('--composition', default=None,
                        help='Filtro de composição (ex: abc, abc_proj)')
    parser.add_argument('--drift',       default=None,
                        choices=['adwin', 'nodrift'],
                        help='Filtro de drift tag')
    parser.add_argument('--seed',        type=int, default=None)
    parser.add_argument('--metric',      default='acc',
                        choices=['acc', 'kappa', 'kappam', 'drifts', 'lat_ms'],
                        help='Métrica para o pivot (padrão: acc)')
    parser.add_argument('--full',        action='store_true',
                        help='Exibe tabela completa linha a linha')
    args = parser.parse_args()

    df = load_results(args.folder, composition=args.composition,
                      drift=args.drift, seed=args.seed)

    if df.empty:
        print(f"Nenhum resultado soft_reset encontrado em '{args.folder}'.")
        exit(0)

    print(f"\nArquivos carregados: {len(df)}")
    print(f"Datasets    : {sorted(df['dataset'].unique())}")
    print(f"Composições : {sorted(df['composition'].unique())}")
    print(f"n_reset     : {sorted(df['n_reset'].unique())}")
    print(f"Drift tags  : {sorted(df['drift'].unique())}")

    tabela_pivot(df, metric=args.metric)
    tabela_drift_vs_nodrift(df)

    if args.full:
        tabela_principal(df)
