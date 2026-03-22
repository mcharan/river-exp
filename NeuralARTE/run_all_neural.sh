#!/bin/bash
# ==============================================================================
# SCRIPT DE DISPARO PARALELO — Neural ARTE (deep-river)
# ==============================================================================

PYTHON="/home/marcelo.charan1/anaconda3/envs/deep-river-demo/bin/python"
SCRIPT_DIR="/home/marcelo.charan1/Documents/river-exp"
SCRIPT="$SCRIPT_DIR/NeuralARTE/neural_arte.py"

# Ajuste conforme necessário
N_MODELS=30
SEED=123456789
LAMBDA=6
WINDOW=500
DATASETS_PATH="/home/marcelo.charan1/Documents/moa/AdaptiveRandomTreeEnsemble/datasets"
LOGS_DIR="$SCRIPT_DIR/results/logs"

mkdir -p "$LOGS_DIR"

# Datasets divididos entre as duas GPUs
# GPU 0: datasets menores/médios
gpu0_datasets=(
    "electricity"
    "outdoor"
    "ozone"
    "shuttle"
    "keystroke"
    "rialto"
    "gmsc"
    "gassensor"
    "covtype"
)

# GPU 1: datasets maiores
gpu1_datasets=(
    "airlines"
    "sea_a"
    "sea_g"
    "led_a"
    "led_g"
    "agrawal_a"
    "agrawal_g"
    "rbf_f"
    "rbf_m"
)

echo "Disparando Neural ARTE em paralelo (n_models=$N_MODELS, seed=$SEED)..."
echo "GPU 0: ${gpu0_datasets[*]}"
echo "GPU 1: ${gpu1_datasets[*]}"
echo ""

for ds in "${gpu0_datasets[@]}"; do
    LOG="$LOGS_DIR/neural_${ds}.log"
    echo "GPU 0 — $ds"
    screen -dmS "neural_$ds" bash -c "
        cd $SCRIPT_DIR
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --dataset $ds --seed $SEED --n_models $N_MODELS --lambda_val $LAMBDA --window $WINDOW --datasets_path $DATASETS_PATH > $LOG 2>&1
    "
    sleep 2
done

for ds in "${gpu1_datasets[@]}"; do
    LOG="$LOGS_DIR/neural_${ds}.log"
    echo "GPU 1 — $ds"
    screen -dmS "neural_$ds" bash -c "
        cd $SCRIPT_DIR
        CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT --dataset $ds --seed $SEED --n_models $N_MODELS --lambda_val $LAMBDA --window $WINDOW --datasets_path $DATASETS_PATH > $LOG 2>&1
    "
    sleep 2
done

echo ""
echo "Todos os experimentos disparados!"
echo "Use 'screen -ls' para ver as sessões ativas."
echo "Logs em tempo real: tail -f $LOGS_DIR/neural_<dataset>.log"
echo "Resultados CSV em: $SCRIPT_DIR/results/neural/"
