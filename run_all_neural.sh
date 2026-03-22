#!/bin/bash
# ==============================================================================
# SCRIPT DE DISPARO PARALELO — Neural ARTE (deep-river)
# ==============================================================================

PYTHON="/home/marcelo.charan1/anaconda3/envs/deep-river-demo/bin/python"
SCRIPT_DIR="/home/marcelo.charan1/Documents/river-exp"
SCRIPT="$SCRIPT_DIR/neural_arte.py"

# Ajuste conforme necessário
N_MODELS=30
SEED=123456789
LAMBDA=6
WINDOW=500

# Datasets a executar (remova ou comente os que não quiser)
datasets=(
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

echo "Disparando Neural ARTE em paralelo (n_models=$N_MODELS, seed=$SEED)..."

for ds in "${datasets[@]}"; do
    echo "Iniciando screen: neural_$ds"
    screen -dmS "neural_$ds" bash -c "
        cd $SCRIPT_DIR
        $PYTHON $SCRIPT --dataset $ds --seed $SEED --n_models $N_MODELS --lambda_val $LAMBDA --window $WINDOW
        exec bash
    "
    sleep 2
done

echo ""
echo "Todos os experimentos disparados!"
echo "Use 'screen -ls' para ver as sessões."
echo "Use 'screen -r neural_<dataset>' para acompanhar."
echo "Resultados salvos em: $SCRIPT_DIR/results/neural/"
