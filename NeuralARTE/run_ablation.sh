#!/bin/bash
# ==============================================================================
# ABLATION: tamanho do ensemble (n_models) + composição abc_extended
# ==============================================================================
# GPU 0: abc com n_models = 10, 60
# GPU 1: abc com n_models = 100  +  abc_extended com n_models = 30
# Datasets: bateria completa (todos os ARFFs disponíveis)
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

# Datasets completos
ALL_DATASETS=(
    "electricity"
    "outdoor"
    "ozone"
    "shuttle"
    "keystroke"
    "rialto"
    "gmsc"
    "gassensor"
    "covtype"
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

echo "============================================================"
echo " ABLATION NeuralARTE"
echo " GPU 0: abc n=10, abc n=60"
echo " GPU 1: abc n=100, abc_extended n=30"
echo "============================================================"
echo ""

# --- GPU 0: abc n=10 ---
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

# --- GPU 0: abc n=60 (após os n=10 terminarem via fila no screen) ---
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

# --- GPU 1: abc n=100 ---
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

# --- GPU 1: abc_extended n=30 ---
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

echo ""
echo "Todos os experimentos disparados!"
echo "Acompanhe: tail -f $LOGS_DIR/neural_<variante>_<dataset>.log"
echo "Resultados CSV: $SCRIPT_DIR/results/neural/"
