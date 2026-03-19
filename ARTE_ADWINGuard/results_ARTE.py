import pandas as pd
import glob
import os

def analisar_resultados(folder_path="results/original"):
    # Lista todos os CSVs na pasta
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # print(f"{'DATASET':<15} | {'ACC FINAL':<9} | {'KAPPA':<6} | {'KAPPA_M':<6} | {'DRIFTS':<6} | {'LATENCIA (ms)':<13} | {'RAM MAX (MB)':<12} | {'TEMPO TOTAL (min)'}")
    print(f"{'DATASET':<15} | {'ACC FINAL':<9} | {'KAPPA':<6} | {'KAPPA_M':<6} | {'DRIFTS':<6} | {'RAM MAX (MB)':<12} | {'TEMPO TOTAL (min)'}")
    print("-" * 100)

    resultados = []

    for f in files:
        try:
            df = pd.read_csv(f)
            
            # Pega a última linha (Estado Final)
            last_row = df.iloc[-1]
            dataset_name = last_row['Dataset']
            
            # 1. Acurácia e Kappa Finais (Última linha)
            final_acc = last_row['Accuracy'] * 100
            final_kappa = last_row['Kappa']
            final_kappam = last_row['Kappa_M']
            total_drifts = last_row['Drifts']
            
            # 2. Latência Média (Média da coluna toda)
            avg_latency = df['Latencia_ms'].mean()
            
            # 3. RAM Máxima (Pico de memória)
            max_ram = df['RAM_MB'].max()
            
            # 4. Tempo Total (Baseado no timestamp se disponível ou cálculo simples)
            # Estimativa simples: soma dos tempos de loop ou delta do timestamp inicial/final
            # Aqui vamos estimar pelo número de linhas * intervalo de log * latencia? 
            # Melhor: Se tiver coluna 'Time', faz delta. Se não, usa latencia.
            total_time_min = 0
            if 'Time' in df.columns:
                t_start = pd.to_datetime(df.iloc[0]['Time'], format='%H:%M:%S')
                t_end = pd.to_datetime(last_row['Time'], format='%H:%M:%S')
                # Ajuste para virada de dia (se t_end < t_start, somar 24h)
                if t_end < t_start:
                    t_end += pd.Timedelta(days=1)
                total_time_min = (t_end - t_start).total_seconds() / 60
            
            resultados.append({
                'dataset': dataset_name,
                'acc': final_acc,
                'kappa': final_kappa,
                'kappam': final_kappam,
                'drifts': total_drifts,
                'latencia': avg_latency,
                'ram': max_ram,
                'tempo': total_time_min
            })

            
            # Análise de Drifts para Sintéticos (Snapshot a cada 250k)
            # Se o dataset for grande (>500k), mostra parciais
            if len(df) * 2000 > 300000: # Assumindo log a cada 2000 inst
                 pass # Pode implementar breakdown aqui se quiser
                 
        except Exception as e:
            print(f"Erro lendo {os.path.basename(f)}: {e}")

    resultados_ordenados = sorted(resultados, key=lambda k: k['dataset'].lower())

    # 4. Agora sim, imprimimos a tabela bonitona e em ordem!
    for r in resultados_ordenados:
        print(f"{r['dataset']:<15} | {r['acc']:05.2f}%    | {r['kappa']:.3f}  | {r['kappam']:.3f}   | {r['drifts']:<6} | {r['ram']:<12.3f} | {r['tempo']:.2f}")

if __name__ == "__main__":
    # Ajuste o caminho para onde seus CSVs estão
    analisar_resultados("results/original")
