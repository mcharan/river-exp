#!/bin/bash
# ==============================================================================
# ABLATION: tamanho do ensemble (n_models) + composição abc_extended
# ==============================================================================
# Onda 1 (executar primeiro):
#   GPU 0: abc n=10   (18 datasets em paralelo)
#   GPU 1: abc_extended n=30  (18 datasets em paralelo)
#
# Onda 2 (executar após onda 1 terminar):
#   GPU 0: abc n=60   (18 datasets em paralelo)
#   GPU 1: abc n=100  (18 datasets em paralelo)
#
# Uso:
#   bash run_ablation.sh 1     # dispara onda 1
#   bash run_ablation.sh 2     # dispara onda 2
#   bash run_ablation.sh auto  # dispara onda 1 e inicia onda 2 automaticamente
# ==============================================================================

PYTHON="/home/marcelo.charan1/.conda/envs/deep-river-demo/bin/python"
SCRIPT_DIR="/home/marcelo.charan1/Documents/river-exp"
SCRIPT="$SCRIPT_DIR/NeuralARTE/neural_arte.py"

SEED=123456789
LAMBDA=6
WINDOW=500
DATASETS_PATH="/home/marcelo.charan1/Documents/moa/AdaptiveRandomTreeEnsemble/datasets"
LOGS_DIR="$SCRIPT_DIR/results/logs"

mkdir -p "$LOGS_DIR"
mkdir -p "$SCRIPT_DIR/results/neural"

ALL_DATASETS=(
    "electricity" "outdoor" "ozone" "shuttle" "keystroke"
    "rialto" "gmsc" "gassensor" "covtype" "airlines"
    "sea_a" "sea_g" "led_a" "led_g"
    "agrawal_a" "agrawal_g" "rbf_f" "rbf_m"
)

WAVE=${1:-1}

if [ "$WAVE" = "1" ]; then
    echo "============================================================"
    echo " ONDA 1"
    echo " GPU 0: abc n=10  |  GPU 1: abc_extended n=30"
    echo "============================================================"

    for ds in "${ALL_DATASETS[@]}"; do
        LOG="$LOGS_DIR/neural_abc10_${ds}.log"
        screen -dmS "abc10_$ds" bash -c "
            cd $SCRIPT_DIR
            CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
                --dataset $ds --seed $SEED --n_models 10 \
                --lambda_val $LAMBDA --window $WINDOW \
                --datasets_path $DATASETS_PATH \
                --composition abc \
                > $LOG 2>&1
        "
        sleep 1
    done
    echo "GPU 0 — abc n=10: ${#ALL_DATASETS[@]} experimentos disparados"

    for ds in "${ALL_DATASETS[@]}"; do
        LOG="$LOGS_DIR/neural_abcext_${ds}.log"
        screen -dmS "abcext_$ds" bash -c "
            cd $SCRIPT_DIR
            CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
                --dataset $ds --seed $SEED --n_models 30 \
                --lambda_val $LAMBDA --window $WINDOW \
                --datasets_path $DATASETS_PATH \
                --composition abc_extended \
                > $LOG 2>&1
        "
        sleep 1
    done
    echo "GPU 1 — abc_extended n=30: ${#ALL_DATASETS[@]} experimentos disparados"

elif [ "$WAVE" = "2" ]; then
    echo "============================================================"
    echo " ONDA 2"
    echo " GPU 0: abc n=60  |  GPU 1: abc n=100"
    echo "============================================================"

    for ds in "${ALL_DATASETS[@]}"; do
        LOG="$LOGS_DIR/neural_abc60_${ds}.log"
        screen -dmS "abc60_$ds" bash -c "
            cd $SCRIPT_DIR
            CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
                --dataset $ds --seed $SEED --n_models 60 \
                --lambda_val $LAMBDA --window $WINDOW \
                --datasets_path $DATASETS_PATH \
                --composition abc \
                > $LOG 2>&1
        "
        sleep 1
    done
    echo "GPU 0 — abc n=60: ${#ALL_DATASETS[@]} experimentos disparados"

    for ds in "${ALL_DATASETS[@]}"; do
        LOG="$LOGS_DIR/neural_abc100_${ds}.log"
        screen -dmS "abc100_$ds" bash -c "
            cd $SCRIPT_DIR
            CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
                --dataset $ds --seed $SEED --n_models 100 \
                --lambda_val $LAMBDA --window $WINDOW \
                --datasets_path $DATASETS_PATH \
                --composition abc \
                > $LOG 2>&1
        "
        sleep 1
    done
    echo "GPU 1 — abc n=100: ${#ALL_DATASETS[@]} experimentos disparados"

elif [ "$WAVE" = "auto" ]; then
    echo "============================================================"
    echo " MODO AUTO: onda 1 → aguarda → onda 2"
    echo "============================================================"

    # Dispara onda 1
    bash "$0" 1

    echo ""
    echo "Aguardando onda 1 terminar (verificando a cada 2 min)..."
    LOG_MONITOR="$LOGS_DIR/ablation_monitor.log"
    echo "[$(date '+%H:%M:%S')] Onda 1 disparada. Monitorando..." | tee "$LOG_MONITOR"

    # Monitora até todas as sessões abc10_* e abcext_* encerrarem
    while screen -ls | grep -qE "abc10_|abcext_"; do
        ATIVOS=$(screen -ls | grep -cE "abc10_|abcext_")
        echo "[$(date '+%H:%M:%S')] Onda 1: $ATIVOS sessões ativas..." | tee -a "$LOG_MONITOR"
        sleep 120
    done

    echo "[$(date '+%H:%M:%S')] Onda 1 concluída. Disparando onda 2..." | tee -a "$LOG_MONITOR"

    # Dispara onda 2
    bash "$0" 2

    echo "[$(date '+%H:%M:%S')] Onda 2 disparada." | tee -a "$LOG_MONITOR"

else
    echo "Uso: bash run_ablation.sh 1     (onda 1)"
    echo "     bash run_ablation.sh 2     (onda 2)"
    echo "     bash run_ablation.sh auto  (sequência automática)"
    exit 1
fi

echo ""
echo "Acompanhe: tail -f $LOGS_DIR/neural_<variante>_<dataset>.log"
echo "Verifique sessões ativas: screen -ls"
echo "Resultados CSV: $SCRIPT_DIR/results/neural/"
