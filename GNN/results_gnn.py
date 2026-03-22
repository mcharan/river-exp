"""
Análise comparativa dos resultados GNN-ARTE vs NeuralARTE baseline.

Lê todos os CSVs de results/gnn/ e apresenta tabela comparativa:
  Dataset | Variante | Acc | Kappa | KappaM | Drifts | Lat(ms) | RAM(MB) | Tempo(min)
"""

import pandas as pd
import glob
import os


def analisar_resultados(folder_path="results/gnn"):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not files:
        print(f"Nenhum CSV encontrado em '{folder_path}'")
        return

    print(f"\n{'DATASET':<15} | {'VARIANTE':<15} | {'ACC':<8} | {'KAPPA':<6} | "
          f"{'KAPPA_M':<7} | {'DRIFTS':<6} | {'LAT(ms)':<8} | {'RAM(MB)':<8} | {'TEMPO(min)'}")
    print("-" * 110)

    rows = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            last = df.iloc[-1]

            dataset  = last.get("Dataset", os.path.basename(f))
            acc      = last.get("Accuracy", float("nan")) * 100
            kappa    = last.get("Kappa",    float("nan"))
            kappa_m  = last.get("KappaM",   float("nan"))
            drifts   = last.get("Drifts",   float("nan"))
            lat      = df["Latencia_ms"].mean()
            ram      = df["RAM_MB"].max()

            # extrai variante do nome do arquivo (ex: electricity_metagnn_s123456789.csv)
            base = os.path.splitext(os.path.basename(f))[0]
            parts = base.split("_")
            seed_part = next((p for p in parts if p.startswith("s") and p[1:].isdigit()), "")
            variant_parts = [p for p in parts if p != str(dataset) and p != seed_part]
            variant = "_".join(variant_parts) if variant_parts else "unknown"

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

            print(f"{str(dataset):<15} | {variant:<15} | {acc:05.2f}%  | {kappa:.3f}  | "
                  f"{kappa_m:.3f}   | {int(drifts):<6} | {lat:.3f}ms  | {ram:.1f}      | {total_time:.1f}")

            rows.append({
                "dataset": dataset, "variant": variant,
                "acc": acc, "kappa": kappa, "kappa_m": kappa_m,
                "drifts": drifts, "lat_ms": lat, "ram_mb": ram, "time_min": total_time
            })

        except Exception as e:
            print(f"Erro lendo {os.path.basename(f)}: {e}")

    # Pivot comparativo: baseline vs metagnn
    if rows:
        print("\n--- Ganho Meta-GNN vs Baseline (por dataset) ---")
        summary = pd.DataFrame(rows)
        for ds in summary["dataset"].unique():
            sub = summary[summary["dataset"] == ds]
            base_row = sub[sub["variant"] == "baseline"]
            gnn_row  = sub[sub["variant"].str.startswith("metagnn") &
                           ~sub["variant"].str.contains("knn")]
            if not base_row.empty and not gnn_row.empty:
                delta_acc = gnn_row.iloc[0]["acc"] - base_row.iloc[0]["acc"]
                delta_lat = gnn_row.iloc[0]["lat_ms"] - base_row.iloc[0]["lat_ms"]
                sign = "+" if delta_acc >= 0 else ""
                print(f"  {str(ds):<15} : ΔAcc={sign}{delta_acc:.2f}pp | ΔLat={delta_lat:+.3f}ms")


if __name__ == "__main__":
    analisar_resultados("results/gnn")
