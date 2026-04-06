"""
compare_moa_python.py — Comparação lado a lado: MOA vs Python/River ARTE

Lê os CSVs de resultados do MOA e do Python e exibe duas tabelas:
  Tabela 1 — Acurácia e Kappa (métricas de qualidade)
  Tabela 2 — Recursos (drifts, memória, tempo, latência)

Uso:
    python analysis/compare_moa_python.py
    python analysis/compare_moa_python.py --moa_folder results_moa_baseline --python_folder results
    python analysis/compare_moa_python.py --mw 10        # filtra resultados Python com mw=10
"""

import pandas as pd
import glob
import os
import re
import argparse


# =============================================================================
# LEITURA DOS RESULTADOS DO MOA
# =============================================================================
def load_moa(folder):
    files = glob.glob(os.path.join(folder, "*_moa_results.csv"))
    rows = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            if df.empty:
                continue
            last = df.iloc[-1]
            dataset = os.path.basename(f).replace("_moa_results.csv", "")

            acc    = float(last.get('classifications correct (percent)', float('nan')))
            kappa  = float(last.get('Kappa Statistic (percent)',         float('nan'))) / 100.0
            kappam = float(last.get('Kappa M Statistic (percent)',       float('nan'))) / 100.0
            time_s = float(last.get('evaluation time (cpu seconds)',     float('nan')))

            # '[avg] model serialized size (bytes)': tamanho do modelo serializado em disco,
            # não a ocupação de heap RAM. Serve como proxy da complexidade do modelo.
            model_bytes = float(last.get('[avg] model serialized size (bytes)', float('nan')))
            model_mb = abs(model_bytes) / (1024 * 1024) if not pd.isna(model_bytes) else float('nan')

            rows.append({
                'dataset':       dataset,
                'moa_acc':       acc,
                'moa_kappa':     kappa,
                'moa_kappam':    kappam,
                'moa_time_s':    time_s,
                'moa_model_mb':  model_mb,
            })
        except Exception as e:
            print(f"[ERRO MOA] {os.path.basename(f)}: {e}")

    return pd.DataFrame(rows)


# =============================================================================
# LEITURA DOS RESULTADOS DO PYTHON
# Padrão de arquivo: ARTE_CPU_{dataset}_mw{mw}_s{seed}_{YYYYMMDD_HHMMSS}.csv
# =============================================================================
def parse_python_filename(filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]
    m = re.match(r'ARTE_CPU_(.+)_mw(\d+)_s(\d+)_(\d{8}_\d{6})$', base)
    if not m:
        return None
    return {
        'dataset':   m.group(1),
        'mw':        int(m.group(2)),
        'seed':      int(m.group(3)),
        'timestamp': m.group(4),
    }


def load_python(folder, mw=None, seed=123456789):
    files = glob.glob(os.path.join(folder, "ARTE_CPU_*.csv"))
    rows = []
    for f in sorted(files):
        meta = parse_python_filename(f)
        if meta is None:
            continue
        if seed is not None and meta['seed'] != seed:
            continue
        if mw is not None and meta['mw'] != mw:
            continue
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            last = df.iloc[-1]

            acc    = float(last.get('Accuracy', float('nan'))) * 100
            kappa  = float(last.get('Kappa',    float('nan')))
            kappam = float(last.get('KappaM',   float('nan')))
            drifts = last.get('Drifts', float('nan'))
            ram_mb = float(last.get('RAM_MB',   float('nan')))
            lat_ms = df['Latencia_ms'].mean() if 'Latencia_ms' in df.columns else float('nan')

            total_time_min = float('nan')
            if 'Time' in df.columns:
                try:
                    t0 = pd.to_datetime(df.iloc[0]['Time'], format='%H:%M:%S')
                    t1 = pd.to_datetime(last['Time'],       format='%H:%M:%S')
                    if t1 < t0:
                        t1 += pd.Timedelta(days=1)
                    total_time_min = (t1 - t0).total_seconds() / 60
                except Exception:
                    pass

            rows.append({
                'dataset':      meta['dataset'],
                'mw':           meta['mw'],
                'seed':         meta['seed'],
                'timestamp':    meta['timestamp'],
                'py_acc':       acc,
                'py_kappa':     kappa,
                'py_kappam':    kappam,
                'py_drifts':    int(drifts) if not pd.isna(drifts) else None,
                'py_ram_mb':    ram_mb,
                'py_lat_ms':    lat_ms,
                'py_time_min':  total_time_min,
            })
        except Exception as e:
            print(f"[ERRO Python] {os.path.basename(f)}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Múltiplas rodadas para o mesmo dataset → mantém a mais recente
    df = df.sort_values('timestamp').groupby('dataset').last().reset_index()
    return df


# =============================================================================
# TABELA 1 — Acurácia e Kappa
# =============================================================================
def tabela_acuracia(merged):
    print(f"\n{'='*90}")
    print(f" TABELA 1 — ACURÁCIA E KAPPA")
    print(f" Δ = Python − MOA  (positivo = Python melhor)")
    print(f"{'='*90}")

    t = merged.sort_values('dataset')

    header = (
        f"{'DATASET':<15} | {'MOA Acc':>8} | {'Py Acc':>8} | {'Δ Acc':>7} | "
        f"{'MOA κ':>7} | {'Py κ':>7} | {'Δ κ':>6} | "
        f"{'MOA κM':>7} | {'Py κM':>7} | {'Δ κM':>6}"
    )
    print(header)
    print("-" * len(header))

    for _, r in t.iterrows():
        def fmt_delta(v):
            return f"{v:+.2f}" if not pd.isna(v) else "  N/A"

        moa_acc_s    = f"{r['moa_acc']:.2f}%"    if not pd.isna(r['moa_acc'])    else "   N/A"
        py_acc_s     = f"{r['py_acc']:.2f}%"     if not pd.isna(r['py_acc'])     else "   N/A"
        moa_kappa_s  = f"{r['moa_kappa']:.3f}"   if not pd.isna(r['moa_kappa'])  else "  N/A"
        py_kappa_s   = f"{r['py_kappa']:.3f}"    if not pd.isna(r['py_kappa'])   else "  N/A"
        moa_kappam_s = f"{r['moa_kappam']:.3f}"  if not pd.isna(r['moa_kappam']) else "  N/A"
        py_kappam_s  = f"{r['py_kappam']:.3f}"   if not pd.isna(r['py_kappam'])  else "  N/A"

        print(
            f"{r['dataset']:<15} | {moa_acc_s:>8} | {py_acc_s:>8} | {fmt_delta(r['delta_acc']):>7} | "
            f"{moa_kappa_s:>7} | {py_kappa_s:>7} | {fmt_delta(r['delta_kappa']):>6} | "
            f"{moa_kappam_s:>7} | {py_kappam_s:>7} | {fmt_delta(r['delta_kappam']):>6}"
        )

    # Resumo
    valid = merged.dropna(subset=['delta_acc'])
    if not valid.empty:
        mean_d = valid['delta_acc'].mean()
        sign   = "+" if mean_d >= 0 else ""
        print(f"\n  Média Δ Acc  : {sign}{mean_d:.2f}pp")
        print(f"  Média Δ κ    : {valid['delta_kappa'].mean():+.3f}")
        print(f"  Média Δ κM   : {valid['delta_kappam'].mean():+.3f}")
        pos = (valid['delta_acc'] > 0).sum()
        neg = (valid['delta_acc'] < 0).sum()
        eq  = (valid['delta_acc'] == 0).sum()
        print(f"  Python melhor: {pos} | MOA melhor: {neg} | Empate: {eq}  (de {len(valid)} datasets)")


# =============================================================================
# TABELA 2 — Recursos
# =============================================================================
def tabela_recursos(merged):
    print(f"\n{'='*90}")
    print(f" TABELA 2 — RECURSOS")
    print(f" MOA Mdl(MB) = tamanho serializado do modelo (proxy, não heap RAM)")
    print(f" Py  RAM(MB) = memória RSS do processo Python (psutil)")
    print(f"{'='*90}")

    t = merged.sort_values('dataset')

    header = (
        f"{'DATASET':<15} | {'Drifts':>6} | {'MOA Mdl(MB)':>11} | {'Py RAM(MB)':>10} | "
        f"{'MOA t(s)':>9} | {'Py t(min)':>9} | {'Py Lat(ms)':>10}"
    )
    print(header)
    print("-" * len(header))

    for _, r in t.iterrows():
        def fv(v, fmt):
            return format(v, fmt) if not pd.isna(v) else "N/A".rjust(len(format(0, fmt)))

        drifts_s = str(int(r['py_drifts'])) if r.get('py_drifts') is not None else "N/A"
        print(
            f"{r['dataset']:<15} | {drifts_s:>6} | {fv(r['moa_model_mb'], '11.2f')} | "
            f"{fv(r['py_ram_mb'], '10.1f')} | {fv(r['moa_time_s'], '9.1f')} | "
            f"{fv(r['py_time_min'], '9.1f')} | {fv(r['py_lat_ms'], '10.3f')}"
        )


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparação MOA vs Python/River ARTE')
    parser.add_argument('--moa_folder',    default='results_moa_baseline',
                        help='Pasta com CSVs do MOA (padrão: results_moa_baseline)')
    parser.add_argument('--python_folder', default='results',
                        help='Pasta com CSVs do Python (padrão: results)')
    parser.add_argument('--mw',  type=int, default=None,
                        help='Filtro min_window_length nos resultados Python (ex: --mw 10)')
    parser.add_argument('--seed', type=int, default=123456789,
                        help='Filtro seed nos resultados Python (padrão: 123456789)')
    args = parser.parse_args()

    moa = load_moa(args.moa_folder)
    py  = load_python(args.python_folder, mw=args.mw, seed=args.seed)

    if moa.empty:
        print(f"Nenhum resultado MOA encontrado em '{args.moa_folder}'.")
        exit(0)
    if py.empty:
        print(f"Nenhum resultado Python encontrado em '{args.python_folder}'.")
        mw_hint = f" --mw {args.mw}" if args.mw else ""
        print(f"  Verifique o padrão de nome: ARTE_CPU_{{dataset}}_mw{{mw}}_s{{seed}}_{{ts}}.csv")
        print(f"  Ou filtre com: --mw N{mw_hint}")
        exit(0)

    moa_datasets = set(moa['dataset'])
    py_datasets  = set(py['dataset'])

    print(f"\nMOA   : {len(moa)} datasets — {sorted(moa_datasets)}")
    print(f"Python: {len(py)}  datasets — {sorted(py_datasets)}")

    only_moa = sorted(moa_datasets - py_datasets)
    only_py  = sorted(py_datasets  - moa_datasets)
    if only_moa:
        print(f"\n  [AVISO] Sem resultado Python para: {only_moa}")
    if only_py:
        print(f"  [AVISO] Sem resultado MOA  para: {only_py}")

    merged = pd.merge(moa, py, on='dataset', how='outer')
    merged['delta_acc']    = merged['py_acc']    - merged['moa_acc']
    merged['delta_kappa']  = merged['py_kappa']  - merged['moa_kappa']
    merged['delta_kappam'] = merged['py_kappam'] - merged['moa_kappam']

    mw_info = f"mw={args.mw}" if args.mw else "mw=qualquer"
    print(f"  Filtro Python: seed={args.seed}, {mw_info}")

    tabela_acuracia(merged)
    tabela_recursos(merged)
