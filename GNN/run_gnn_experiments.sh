#!/bin/bash
# ==============================================================================
# SCRIPT DE DISPARO — GNN-ARTE (Fase 1: Elec2 + outdoor como prova de conceito)
# ==============================================================================
# Roda 3 variantes por dataset:
#   1. baseline   — votação majoritária (sem Meta-GNN)
#   2. metagnn    — MetaGNN full graph
#   3. metagnn_knn — MetaGNN knn graph (k=5)
# ==============================================================================

PYTHON="/home/marcelo.charan1/.conda/envs/deep-river-demo/bin/python"
SCRIPT_DIR="/home/marcelo.charan1/Documents/river-exp"
SCRIPT="$SCRIPT_DIR/GNN/gnn_arte.py"

N_MODELS=30
SEED=123456789
LAMBDA=6
WINDOW=500
DATASETS_PATH="/home/marcelo.charan1/Documents/moa/AdaptiveRandomTreeEnsemble/datasets"
LOGS_DIR="$SCRIPT_DIR/results/logs"

mkdir -p "$LOGS_DIR"
mkdir -p "$SCRIPT_DIR/results/gnn"

# Datasets para prova de conceito (rápidos)
POC_DATASETS=(
    "electricity"
    "outdoor"
    "sea_g"
)

echo "Disparando GNN-ARTE (prova de conceito)..."
echo "Datasets: ${POC_DATASETS[*]}"
echo ""

for ds in "${POC_DATASETS[@]}"; do
    for variant in "baseline" "metagnn" "metagnn_knn"; do
        LOG="$LOGS_DIR/gnn_${ds}_${variant}.log"
        TAG="gnn_${ds}_${variant}"

        if [ "$variant" = "baseline" ]; then
            EXTRA_ARGS="--no_metagnn"
        elif [ "$variant" = "metagnn" ]; then
            EXTRA_ARGS="--graph_type full"
        else
            EXTRA_ARGS="--graph_type knn"
        fi

        echo "  [$TAG]"
        screen -dmS "$TAG" bash -c "
            cd $SCRIPT_DIR
            CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
                --dataset $ds \
                --seed $SEED \
                --n_models $N_MODELS \
                --lambda_val $LAMBDA \
                --window $WINDOW \
                --datasets_path $DATASETS_PATH \
                --gnn_hidden 64 \
                --gnn_update 10 \
                --gnn_heads 4 \
                $EXTRA_ARGS \
                > $LOG 2>&1
        "
        sleep 2
    done
done

echo ""
echo "Experimentos disparados!"
echo "Logs: tail -f $LOGS_DIR/gnn_<dataset>_<variant>.log"
echo "Resultados CSV: $SCRIPT_DIR/results/gnn/"
echo ""
echo "Use 'screen -ls' para ver sessões ativas."
