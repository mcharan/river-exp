"""
Comparação dos resultados da ablação do NeuralARTE.

Lê todos os CSVs de results/neural/ e agrupa por (dataset, composição, n_models),
exibindo tabelas comparativas de acurácia, kappa, drifts e latência.

Uso:
    python analysis/compare_ablation.py
    python analysis/compare_ablation.py --folder results/neural --metric acc
"""

import pandas as pd
import glob
import os
import re
import argparse


# =============================================================================
# PARSING DO NOME DO ARQUIVO
# NeuralARTE_{dataset}_{composition}_{drift_tag}_s{seed}_{timestamp}.csv
# =============================================================================
def parse_filename(filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]
    # Remove prefixo
    base = base.replace("NeuralARTE_", "")
    # Extrai seed e timestamp do final
    m = re.search(r'_s(\d+)_(\d{8}_\d{6})$', base)
    if not m:
        return None
    seed = int(m.group(1))
    remainder = base[:m.start()]   # dataset_composition_drift_tag
    # drift_tag é sempre 'adwin' ou 'nodrift'
    for drift_tag in ("nodrift", "adwin"):
        if remainder.endswith(f"_{drift_tag}"):
            composition = remainder[:-(len(drift_tag) + 1)]
            # composition ainda tem dataset no início — remove pelo primeiro '_'
            # mas dataset pode ter '_' (ex: agrawal_a), então não podemos partir simplesmente
            # Os compositions conhecidos:
            known = ["abc_extended", "abc_proj", "current", "abc"]
            dataset = None
            comp = None
            for c in known:
                suffix = f"_{c}"
                if composition.endswith(suffix):
                    dataset = composition[:-(len(c) + 1)]
                    comp = c
                    break
            if dataset and comp:
                return {"dataset": dataset, "composition": comp,
                        "drift": drift_tag, "seed": seed}
    return None


# =============================================================================
# LEITURA DOS CSVs
# =============================================================================
def load_results(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
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

            # n_models: infere pelo Run_ID se disponível, senão marca como desconhecido
            run_id = str(last.get("Run_ID", ""))
            n_models = None
            # tenta extrair do log (não está no CSV) — deixa None por ora

            acc      = last.get("Accuracy", float("nan")) * 100
            kappa    = last.get("Kappa",    float("nan"))
            kappa_m  = last.get("KappaM",   float("nan"))
            drifts   = last.get("Drifts",   float("nan"))
            lat      = df["Latencia_ms"].mean()
            ram      = df["RAM_MB"].max()

            total_time = 0.0
            if "Time" in df.columns:
                try:
                    t0 = pd.to_datetime(df.iloc[0]["Time"], format="%H:%M:%S")
                    t1 = pd.to_datetime(last["Time"],       format="%H:%M:%S")
                    if t1 < t0:
                        t1 += pd.Timedelta(days=1)
                    total_time = (t1 - t0).total_seconds() / 60
                except Exception:
                    pass

            rows.append({
                "dataset":     meta["dataset"],
                "composition": meta["composition"],
                "drift":       meta["drift"],
                "seed":        meta["seed"],
                "acc":         acc,
                "kappa":       kappa,
                "kappa_m":     kappa_m,
                "drifts":      int(drifts) if not pd.isna(drifts) else None,
                "lat_ms":      lat,
                "ram_mb":      ram,
                "time_min":    total_time,
                "file":        os.path.basename(f),
            })
        except Exception as e:
            print(f"[ERRO] {os.path.basename(f)}: {e}")

    return pd.DataFrame(rows)


# =============================================================================
# TABELAS DE COMPARAÇÃO
# =============================================================================
def tabela_por_composicao(df, metric="acc"):
    """Pivot: linhas = dataset, colunas = composição. Mostra a métrica escolhida."""
    metric_label = {"acc": "Acurácia (%)", "kappa": "Kappa", "drifts": "Drifts", "lat_ms": "Lat (ms)"}
    print(f"\n{'='*70}")
    print(f" COMPARAÇÃO POR COMPOSIÇÃO — {metric_label.get(metric, metric)}")
    print(f" (drift=adwin, seed fixo)")
    print(f"{'='*70}")

    sub = df[df["drift"] == "adwin"].copy()
    if sub.empty:
        print("Nenhum resultado com drift=adwin encontrado.")
        return

    pivot = sub.pivot_table(index="dataset", columns="composition",
                            values=metric, aggfunc="mean")
    pivot = pivot.sort_index()

    # Destaca melhor por linha
    print(pivot.round(3).to_string())

    # Δ em relação ao abc (baseline da ablação)
    if "abc" in pivot.columns and len(pivot.columns) > 1:
        print(f"\n--- Δ vs abc ---")
        for col in pivot.columns:
            if col != "abc":
                delta = (pivot[col] - pivot["abc"]).mean()
                sign = "+" if delta >= 0 else ""
                print(f"  {col:<15}: média {sign}{delta:.3f}")


def tabela_drift_vs_nodrift(df):
    """Compara adwin vs nodrift para cada composição."""
    print(f"\n{'='*70}")
    print(f" DRIFT DETECTOR: adwin vs nodrift — Acurácia (%)")
    print(f"{'='*70}")

    pivot = df.pivot_table(index=["dataset", "composition"],
                           columns="drift", values="acc", aggfunc="mean")
    if "adwin" not in pivot.columns or "nodrift" not in pivot.columns:
        print("Faltam resultados de uma das variantes (adwin ou nodrift).")
        return

    pivot["Δ (nodrift−adwin)"] = pivot["nodrift"] - pivot["adwin"]
    print(pivot.round(3).to_string())

    avg_delta = pivot["Δ (nodrift−adwin)"].mean()
    sign = "+" if avg_delta >= 0 else ""
    print(f"\n  Média geral Δ: {sign}{avg_delta:.3f}pp")
    print(f"  Positivo = sem drift melhor | Negativo = com drift melhor")


def tabela_resumo(df):
    """Resumo geral: melhor composição por dataset."""
    print(f"\n{'='*70}")
    print(f" RESUMO: MELHOR COMPOSIÇÃO POR DATASET (adwin, por acurácia)")
    print(f"{'='*70}")

    sub = df[df["drift"] == "adwin"]
    if sub.empty:
        return

    idx = sub.groupby("dataset")["acc"].idxmax()
    best = sub.loc[idx, ["dataset", "composition", "acc", "kappa", "drifts", "lat_ms", "time_min"]]
    best = best.sort_values("dataset")
    print(best.round(3).to_string(index=False))


def tabela_completa(df):
    """Tabela completa linha a linha."""
    print(f"\n{'='*90}")
    print(f" TABELA COMPLETA")
    print(f"{'='*90}")
    cols = ["dataset", "composition", "drift", "acc", "kappa", "kappa_m",
            "drifts", "lat_ms", "ram_mb", "time_min"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values(["dataset", "composition", "drift"])
          .round(3).to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparação da ablação NeuralARTE")
    parser.add_argument("--folder", default="results/neural",
                        help="Pasta com os CSVs (padrão: results/neural)")
    parser.add_argument("--metric", default="acc",
                        choices=["acc", "kappa", "drifts", "lat_ms"],
                        help="Métrica para o pivot de composições (padrão: acc)")
    parser.add_argument("--full", action="store_true",
                        help="Exibe tabela completa linha a linha")
    args = parser.parse_args()

    df = load_results(args.folder)
    if df.empty:
        print(f"Nenhum resultado encontrado em '{args.folder}'.")
        exit(0)

    print(f"\nArquivos carregados: {len(df)}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Composições: {sorted(df['composition'].unique())}")
    print(f"Drift tags: {sorted(df['drift'].unique())}")

    tabela_por_composicao(df, metric=args.metric)
    tabela_drift_vs_nodrift(df)
    tabela_resumo(df)

    if args.full:
        tabela_completa(df)
