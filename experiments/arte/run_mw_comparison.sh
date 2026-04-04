#!/bin/bash
# ==============================================================================
# Comparação ADWIN min_window_length=5 (River padrão) vs =10 (equivalente MOA)
#
# Uso:
#   bash run_mw_comparison.sh                              # full, todos os sintéticos
#   bash run_mw_comparison.sh mini                         # reduced (50k inst)
#   bash run_mw_comparison.sh full rbf_m rbf_f agrawal_a  # full, datasets específicos
#   bash run_mw_comparison.sh mini sea_a sea_g             # mini, datasets específicos
#
# Resultados em: results/arte/ARTE_CPU_{dataset}_mw{5|10}_s{seed}_{ts}.csv
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/experiments/arte/run_experiments.py"
LOGS_DIR="$SCRIPT_DIR/results/logs"

SEED=123456789
N_MODELS=100
WINDOW=500

MODE="${1:-full}"
shift  # remove o argumento de modo; o restante são datasets opcionais

ALL_SYNTHETIC=("rbf_m" "rbf_f" "agrawal_a" "agrawal_g" "led_a" "led_g" "sea_a" "sea_g" "mixed_a" "mixed_g")

if [ "$MODE" = "mini" ]; then
    DATASETS_PATH="${DATASETS_PATH:-/home/charan/moa/aldopaim/AdaptiveRandomTreeEnsemble/datasets/mini}"
else
    DATASETS_PATH="${DATASETS_PATH:-/home/charan/moa/aldopaim/AdaptiveRandomTreeEnsemble/datasets}"
fi

# Se foram passados datasets específicos, usa-os; senão usa todos os sintéticos
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_SYNTHETIC[@]}")
fi

mkdir -p "$LOGS_DIR"
mkdir -p "$SCRIPT_DIR/results/arte"

echo "============================================================"
echo " Bateria mw=5 vs mw=10  |  Modo: $MODE"
echo " Datasets: ${#DATASETS[@]}  |  Path: $DATASETS_PATH"
echo "============================================================"

for ds in "${DATASETS[@]}"; do
    for mw in 5 10; do
        SESSION="arte_${ds}_mw${mw}"
        LOG="$LOGS_DIR/${SESSION}.log"

        # Pula se já existe sessão ativa com esse nome
        if screen -ls | grep -q "$SESSION"; then
            echo "[SKIP] Sessão '$SESSION' já está ativa."
            continue
        fi

        echo "Disparando: $SESSION"
        screen -dmS "$SESSION" bash -c "
            cd $SCRIPT_DIR
            $PYTHON $SCRIPT \
                --dataset $ds \
                --seed $SEED \
                --n_models $N_MODELS \
                --window_size $WINDOW \
                --adwin_min_window $mw \
                --datasets_path $DATASETS_PATH \
                > $LOG 2>&1
        "
        sleep 1
    done
done

echo ""
echo "Disparados: ${#DATASETS[@]} datasets × 2 configurações = $((${#DATASETS[@]} * 2)) experimentos"
echo "Acompanhe: tail -f $LOGS_DIR/arte_<dataset>_mw<5|10>.log"
echo "Sessões:   screen -ls | grep arte_"
