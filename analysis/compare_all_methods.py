#!/usr/bin/env python3
"""
Tabela comparativa de accuracy por dataset × método (melhor config adwin disponível).

Uso:
    python analysis/compare_all_methods.py
    python analysis/compare_all_methods.py --metric kappa_m
    python analysis/compare_all_methods.py --save            # salva CSV + MD em analysis/
"""
import os, csv, re, argparse
from collections import defaultdict

SIZES = {
    'agrawal_a': 1000000, 'agrawal_g': 1000000,
    'led_a':     1000000, 'led_g':     1000000,
    'sea_a':     1000000, 'sea_g':     1000000,
    'rbf_f':     1000000, 'rbf_m':     1000000,
    'mixed_a':   1000000, 'mixed_g':   1000000,
    'covtype':    581012, 'airlines':   539383,
    'electricity': 45312, 'gassensor':  13910,
    'gmsc':       150000, 'keystroke':  20400,
    'outdoor':      4000, 'ozone':       2536,
    'rialto':      82250, 'shuttle':    58000,
}

DS_ORDER = [
    'keystroke', 'ozone', 'outdoor', 'gassensor', 'electricity', 'shuttle',
    'rialto', 'gmsc', 'covtype', 'airlines',
    'sea_a', 'sea_g', 'mixed_a', 'mixed_g',
    'led_a', 'led_g', 'agrawal_a', 'agrawal_g', 'rbf_f', 'rbf_m',
]

# Preference order for adwin configs per method
ARTE_CANDS = ['mw10']
NR_CANDS   = ['abc_proj_adwin_dir', 'abc_adwin_dir']
SR_CANDS   = ['soft_reset_abc_proj_rl1_adwin', 'soft_reset_abc_rl1_adwin',
               'soft_reset_abc_cnn_rl1_adwin', 'soft_reset_current_rl1_adwin']
HB_CANDS   = ['abc_proj_adwin_dir', 'abc_adwin_dir', 'heterogeneous_adwin_dir']


def load_best(folder, prefix, metric_idx=4):
    """
    Loads complete runs (≥98% of expected instances, skipping mini runs).
    Returns dict: (dataset, config) -> (accuracy, kappa_m, n_instances, filename)
    When multiple files match the same key, keeps the one with most instances.
    metric_idx: 4=accuracy, 6=kappa_m (column index in CSV)
    """
    best = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.csv') or not fname.startswith(prefix):
            continue
        stem = fname[len(prefix):-4]
        ds = next((d for d in SIZES if stem.startswith(d + '_')), None)
        if ds is None:
            continue
        config = stem[len(ds) + 1:]
        config = re.sub(r'_s\d{9}_\d{8}_\d{6}$', '', config)
        expected = SIZES[ds]
        path = os.path.join(folder, fname)
        try:
            with open(path) as fh:
                rows = list(csv.reader(fh))
            if len(rows) < 2:
                continue
            last = int(rows[-1][2])
            if last <= 50000 and expected > 50000:
                continue  # mini run
            if last < expected * 0.98:
                continue  # incomplete
            acc = float(rows[-1][4])
            km  = float(rows[-1][6])
            key = (ds, config)
            if key not in best or last > best[key][2]:
                best[key] = (acc, km, last, fname)
        except Exception:
            pass
    return best


def best_adwin(data, ds, candidates):
    for c in candidates:
        if (ds, c) in data:
            return data[(ds, c)]
    return None


def build_table(arte, nr, sr, hb, metric='accuracy'):
    """
    Returns list of dicts with keys: dataset, ARTE, NR, SR, HB, best
    metric: 'accuracy' (index 0) or 'kappa_m' (index 1)
    """
    midx = 0 if metric == 'accuracy' else 1
    rows = []
    for ds in DS_ORDER:
        a = best_adwin(arte, ds, ARTE_CANDS)
        n = best_adwin(nr,   ds, NR_CANDS)
        s = best_adwin(sr,   ds, SR_CANDS)
        h = best_adwin(hb,   ds, HB_CANDS)
        vals = {
            'ARTE': a[midx] if a else None,
            'NR':   n[midx] if n else None,
            'SR':   s[midx] if s else None,
            'HB':   h[midx] if h else None,
        }
        avail = {k: v for k, v in vals.items() if v is not None}
        best_k = max(avail, key=avail.get) if avail else None
        rows.append({'dataset': ds, **vals, 'best': best_k})
    return rows


def print_table(rows, metric='accuracy'):
    label = 'ACCURACY' if metric == 'accuracy' else 'KAPPA_M'
    print(f"\n=== {label} — ADWIN (melhor config disponível por método) ===\n")
    print(f"{'Dataset':<14} {'ARTE':>7} {'NeuralARTE':>10} {'SoftReset':>10} {'HeteroBag':>10}  best")
    print("-" * 65)

    sums = defaultdict(float)
    cnts = defaultdict(int)
    wins = defaultdict(int)

    def fmt(v, k, best_k):
        if v is None:
            return f"{'—':>10}"
        s = f"{v * 100:6.2f}%"
        return f"{'▶' + s:>10}" if k == best_k else f"{s:>10}"

    for r in rows:
        bk = r['best']
        print(f"{r['dataset']:<14}"
              f" {fmt(r['ARTE'], 'ARTE', bk)}"
              f" {fmt(r['NR'],   'NR',   bk)}"
              f" {fmt(r['SR'],   'SR',   bk)}"
              f" {fmt(r['HB'],   'HB',   bk)}"
              f"  {bk or '—'}")
        for k in ('ARTE', 'NR', 'SR', 'HB'):
            if r[k] is not None:
                sums[k] += r[k]
                cnts[k] += 1
                if k == bk:
                    wins[k] += 1

    print("-" * 65)
    print(f"{'Wins':<14}", "".join(f"{wins[k]:>10}" for k in ['ARTE', 'NR', 'SR', 'HB']))
    avgs = []
    for k in ['ARTE', 'NR', 'SR', 'HB']:
        avgs.append(f"{sums[k] / cnts[k] * 100:9.2f}%" if cnts[k] else f"{'—':>10}")
    print(f"{'Avg (avail.)':<14}", "".join(avgs))
    print()


def save_markdown(rows_acc, rows_km, arte, nr, sr, hb, out_path):
    lines = []
    lines.append("# Resultados Preliminares — ADWIN Direcional\n")
    lines.append(f"**Gerado em:** 2026-04-19  "
                 f"|  ARTE={len(arte)}  NeuralARTE={len(nr)}  "
                 f"SoftReset={len(sr)}  HeteroBagging={len(hb)}\n")
    lines.append("**Configs selecionadas (melhor disponível por método):**\n")
    lines.append("- ARTE: `mw10`")
    lines.append("- NeuralARTE: `abc_proj_adwin_dir` → `abc_adwin_dir`")
    lines.append("- SoftReset: `abc_proj_rl1_adwin` → `abc_rl1_adwin` → `abc_cnn_rl1_adwin`")
    lines.append("- HeteroBagging: `abc_proj_adwin_dir` → `abc_adwin_dir` → `heterogeneous_adwin_dir`\n")

    for label, rows in [("Accuracy", rows_acc), ("Kappa_M", rows_km)]:
        lines.append(f"## {label} — melhor config por método\n")
        lines.append("| Dataset | ARTE | NeuralARTE | SoftReset | HeteroBagging | Melhor |")
        lines.append("|---------|-----:|----------:|----------:|-------------:|--------|")
        sums = defaultdict(float); cnts = defaultdict(int); wins = defaultdict(int)
        for r in rows:
            bk = r['best']
            def fmt(v, k):
                if v is None: return "—"
                s = f"{v * 100:.2f}%"
                return f"**{s}**" if k == bk else s
            lines.append(f"| {r['dataset']} | {fmt(r['ARTE'],'ARTE')} | {fmt(r['NR'],'NR')} "
                         f"| {fmt(r['SR'],'SR')} | {fmt(r['HB'],'HB')} | {bk or '—'} |")
            for k in ('ARTE', 'NR', 'SR', 'HB'):
                if r[k] is not None:
                    sums[k] += r[k]; cnts[k] += 1
                    if k == bk: wins[k] += 1
        lines.append(f"| **Wins** | **{wins['ARTE']}** | **{wins['NR']}** "
                     f"| **{wins['SR']}** | **{wins['HB']}** | |")
        avg_row = "| **Avg** |"
        for k in ('ARTE', 'NR', 'SR', 'HB'):
            avg_row += f" {sums[k]/cnts[k]*100:.2f}% |" if cnts[k] else " — |"
        avg_row += " |"
        lines.append(avg_row + "\n")

    lines.append("## Datasets incompletos por método (adwin)\n")
    lines.append("| Dataset | ARTE | NeuralARTE | SoftReset | HeteroBagging |")
    lines.append("|---------|:----:|:----------:|:----------:|:-------------:|")
    for ds in DS_ORDER:
        a = "✓" if best_adwin(arte, ds, ARTE_CANDS) else "**falta**"
        n = "✓" if best_adwin(nr,   ds, NR_CANDS)   else "**falta**"
        s = "✓" if best_adwin(sr,   ds, SR_CANDS)   else "**falta**"
        h = "✓" if best_adwin(hb,   ds, HB_CANDS)   else "**falta**"
        if "**falta**" in (a, n, s, h):
            lines.append(f"| {ds} | {a} | {n} | {s} | {h} |")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Markdown salvo: {out_path}")


def save_csv(arte, nr, sr, hb, out_path):
    with open(out_path, 'w', newline='') as fout:
        w = csv.writer(fout)
        w.writerow(['dataset', 'method', 'config', 'accuracy', 'kappa_m', 'n_instances', 'filename'])
        for (ds, cfg), (acc, km, last, fname) in sorted(arte.items()):
            w.writerow([ds, 'ARTE', cfg, f'{acc:.6f}', f'{km:.6f}', last, fname])
        for (ds, cfg), (acc, km, last, fname) in sorted(nr.items()):
            w.writerow([ds, 'NeuralARTE', cfg, f'{acc:.6f}', f'{km:.6f}', last, fname])
        for (ds, cfg), (acc, km, last, fname) in sorted(sr.items()):
            w.writerow([ds, 'SoftReset', cfg, f'{acc:.6f}', f'{km:.6f}', last, fname])
        for (ds, cfg), (acc, km, last, fname) in sorted(hb.items()):
            w.writerow([ds, 'HeteroBagging', cfg, f'{acc:.6f}', f'{km:.6f}', last, fname])
    print(f"CSV salvo: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Comparativo ARTE × NeuralARTE × SoftReset × HeteroBagging")
    parser.add_argument('--metric', choices=['accuracy', 'kappa_m'], default='accuracy')
    parser.add_argument('--save', action='store_true', help='Salva CSV e Markdown em analysis/')
    parser.add_argument('--results_dir', default='.', help='Raiz do projeto (default: cwd)')
    args = parser.parse_args()

    neural_dir = os.path.join(args.results_dir, 'results/neural')
    arte_dir   = os.path.join(args.results_dir, 'results/arte')

    arte   = load_best(arte_dir,   'ARTE_CPU_')
    neural = load_best(neural_dir, 'NeuralARTE_')
    hb     = load_best(neural_dir, 'HeteroBagging_')
    nr = {k: v for k, v in neural.items() if 'soft_reset' not in k[1] and 'subspace' not in k[1]}
    sr = {k: v for k, v in neural.items() if 'soft_reset' in k[1]}

    print(f"Runs completas — ARTE={len(arte)}  NeuralARTE={len(nr)}  SoftReset={len(sr)}  HeteroBagging={len(hb)}")

    rows = build_table(arte, nr, sr, hb, metric=args.metric)
    print_table(rows, metric=args.metric)

    if args.save:
        analysis_dir = os.path.join(args.results_dir, 'analysis')
        save_csv(arte, nr, sr, hb, os.path.join(analysis_dir, 'results_summary_adwin.csv'))
        rows_acc = build_table(arte, nr, sr, hb, metric='accuracy')
        rows_km  = build_table(arte, nr, sr, hb, metric='kappa_m')
        save_markdown(rows_acc, rows_km, arte, nr, sr, hb,
                      os.path.join(analysis_dir, 'results_summary_adwin.md'))


if __name__ == '__main__':
    main()
