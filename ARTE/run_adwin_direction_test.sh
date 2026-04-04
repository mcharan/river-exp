#!/bin/bash
# ==============================================================================
# Validação do ADWINChangeDetector com checagem de direção (equivalente MOA)
#
# Roda todos os datasets sintéticos com o novo ADWINChangeDetector que só
# dispara quando o erro AUMENTA, replicando o ADWINChangeDetector.java do MOA.
#
# Uso:
#   bash ARTE/run_adwin_direction_test.sh
#   bash ARTE/run_adwin_direction_test.sh mini   # datasets reduzidos (50k)
# ==============================================================================

PYTHON="/home/marcelo.charan1/.conda/envs/deep-river-demo/bin/python"
SCRIPT_DIR="/home/marcelo.charan1/Documents/river-exp"
SCRIPT="$SCRIPT_DIR/ARTE/ARTE-Experiments.py"
LOGS_DIR="$SCRIPT_DIR/results/logs"

SEED=123456789
N_MODELS=100
WINDOW=500
MW=5   # min_window_length — mw=5 é o padrão River; a direção é o novo controle

MODE="${1:-full}"

if [ "$MODE" = "mini" ]; then
    DATASETS_PATH="/home/marcelo.charan1/Documents/moa/AdaptiveRandomTreeEnsemble/datasets/mini"
    RESULTS_DIR="$SCRIPT_DIR/results/adwin_direction/mini"
else
    DATASETS_PATH="/home/marcelo.charan1/Documents/moa/AdaptiveRandomTreeEnsemble/datasets"
    RESULTS_DIR="$SCRIPT_DIR/results/adwin_direction/full"
fi

ALL_DATASETS=(
    "rbf_m" "rbf_f"
    "agrawal_a" "agrawal_g"
    "led_a" "led_g"
    "sea_a" "sea_g"
    "mixed_a" "mixed_g"
)

mkdir -p "$LOGS_DIR"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo " ADWINChangeDetector (direção) — Modo: $MODE"
echo " ${#ALL_DATASETS[@]} datasets | n_models=$N_MODELS | mw=$MW"
echo " Resultados: $RESULTS_DIR"
echo "============================================================"

for ds in "${ALL_DATASETS[@]}"; do
    SESSION="adwindir_${ds}"
    LOG="$LOGS_DIR/${SESSION}.log"

    if screen -ls | grep -q "$SESSION"; then
        echo "[SKIP] '$SESSION' já está ativa."
        continue
    fi

    echo "  Disparando: $SESSION"
    screen -dmS "$SESSION" bash -c "
        cd $SCRIPT_DIR
        $PYTHON $SCRIPT \
            --dataset $ds \
            --seed $SEED \
            --n_models $N_MODELS \
            --window_size $WINDOW \
            --adwin_min_window $MW \
            --datasets_path $DATASETS_PATH \
            > $LOG 2>&1

        # Move o CSV gerado para a pasta dedicada
        mv $SCRIPT_DIR/results/original/ARTE_CPU_${ds}_mw${MW}_s${SEED}_*.csv \
           $RESULTS_DIR/ 2>/dev/null
    "
    sleep 1
done

echo ""
echo "Disparados: ${#ALL_DATASETS[@]} experimentos"
echo "Logs:       tail -f $LOGS_DIR/adwindir_<dataset>.log"
echo "Sessões:    screen -ls | grep adwindir_"
echo "Resultados: $RESULTS_DIR"
