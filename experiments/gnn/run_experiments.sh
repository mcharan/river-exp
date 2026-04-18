#!/bin/bash
# ==============================================================================
# SCRIPT DE DISPARO — GNN-ARTE (Fase 1: Elec2 + outdoor como prova de conceito)
# ==============================================================================
# Roda 3 variantes por dataset:
#   1. baseline   — votação majoritária (sem Meta-GNN)
#   2. metagnn    — MetaGNN full graph
#   3. metagnn_knn — MetaGNN knn graph (k=5)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/src/gnn/gnn_arte.py"

N_MODELS=30
SEED=123456789
LAMBDA=6
WINDOW=500
DATASETS_PATH="${DATASETS_PATH:-/home/charan/moa/aldopaim/AdaptiveRandomTreeEnsemble/datasets}"
LOGS_DIR="$SCRIPT_DIR/results/logs"

mkdir -p "$LOGS_DIR"
mkdir -p "$SCRIPT_DIR/results/gnn"

# Distribuição por GPU:
#   GPU 0: electricity (3 variantes) — dataset mais longo, GPU dedicada
#   GPU 1: sea_g + outdoor (6 variantes)
declare -A DATASET_GPU=(
    ["electricity"]="0"
    ["sea_g"]="1"
    ["outdoor"]="1"
)

echo "Disparando GNN-ARTE (prova de conceito)..."
echo "GPU 0: electricity | GPU 1: sea_g + outdoor"
echo ""

for ds in "electricity" "sea_g" "outdoor"; do
    GPU="${DATASET_GPU[$ds]}"
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

        echo "  [$TAG] → GPU $GPU"
        tmux new-session -d -s "$TAG" bash -c "
            cd $SCRIPT_DIR
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SCRIPT \
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
