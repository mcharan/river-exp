#!/bin/bash
# ==============================================================================
# Execucao em ondas — HeterogeneousOnlineBagging
#
# Uso:
#   bash experiments/neural_arte/run_hetero_bagging.sh
#   bash experiments/neural_arte/run_hetero_bagging.sh --composition abc_proj --wave 5 --gpu 0
#   bash experiments/neural_arte/run_hetero_bagging.sh --no_drift
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/experiments/neural_arte/hetero_bagging.py"
LOGS_DIR="$SCRIPT_DIR/results/logs"

DATASETS_PATH="${DATASETS_PATH:-}"
DATASETS_PATH_REAL="${DATASETS_PATH_REAL:-}"

WAVE_SIZE=3
GPU=0
N_MODELS=30
COMPOSITION="abc"
EXTRA_ARGS=""
DRIFT_TAG="adwin"

while [[ $# -gt 0 ]]; do
    case $1 in
        --wave)        WAVE_SIZE="$2";   shift 2 ;;
        --gpu)         GPU="$2";         shift 2 ;;
        --n_models)    N_MODELS="$2";    shift 2 ;;
        --composition) COMPOSITION="$2"; shift 2 ;;
        --no_drift)    EXTRA_ARGS="$EXTRA_ARGS --no_drift"; DRIFT_TAG="nodrift"; shift ;;
        *) echo "Argumento desconhecido: $1"; exit 1 ;;
    esac
done

datasets=(
    "keystroke"
    "ozone"
    "outdoor"
    "gassensor"
    "electricity"
    "shuttle"
    "rialto"
    "gmsc"
    "covtype"
    "airlines"
    "sea_a"
    "sea_g"
    "led_a"
    "led_g"
    "agrawal_a"
    "agrawal_g"
    "mixed_a"
    "mixed_g"
    "rbf_f"
    "rbf_m"
)

DATASETS_ARG=""
if [ -n "$DATASETS_PATH" ]; then
    if [ ! -d "$DATASETS_PATH" ]; then
        echo "[ERRO] DATASETS_PATH nao existe: $DATASETS_PATH"
        exit 1
    fi
    DATASETS_ARG="--datasets_path $DATASETS_PATH"
else
    echo "[AVISO] DATASETS_PATH nao definido — usando o default do codigo."
fi

if [ -n "$DATASETS_PATH_REAL" ]; then
    if [ ! -d "$DATASETS_PATH_REAL" ]; then
        echo "[ERRO] DATASETS_PATH_REAL nao existe: $DATASETS_PATH_REAL"
        exit 1
    fi
    DATASETS_ARG="$DATASETS_ARG --datasets_path_real $DATASETS_PATH_REAL"
else
    echo "[AVISO] DATASETS_PATH_REAL nao definido — datasets reais usarao o mesmo caminho."
fi

mkdir -p "$LOGS_DIR" "$SCRIPT_DIR/results/neural"

total=${#datasets[@]}
wave=0

echo "============================================================"
echo " HeteroBagging | composition=$COMPOSITION | ondas=$WAVE_SIZE | GPU=$GPU"
echo " n_models=$N_MODELS | $total datasets | Logs: $LOGS_DIR"
echo "============================================================"

i=0
while [ $i -lt $total ]; do
    wave=$((wave + 1))
    echo ""
    echo "--- Onda $wave ---"

    sessions=()
    for (( j=0; j<WAVE_SIZE && i<total; j++, i++ )); do
        ds="${datasets[$i]}"
        session="heteroB_${COMPOSITION}_${DRIFT_TAG}_${ds}"
        LOG="$LOGS_DIR/${session}.log"

        if tmux has-session -t "$session" 2>/dev/null; then
            echo "  [SKIP] $session ja esta ativa"
        else
            echo "  Disparando: $session"
            CUDA_ENV=$([ "$GPU" -lt 0 ] && echo "" || echo "$GPU")
            tmux new-session -d -s "$session" bash -c "
                cd $SCRIPT_DIR
                CUDA_VISIBLE_DEVICES=$CUDA_ENV $PYTHON $SCRIPT \
                    --dataset $ds \
                    --composition $COMPOSITION \
                    --n_models $N_MODELS \
                    --seed 123456789 \
                    --lambda_val 6 \
                    --window 500 \
                    --gpu $GPU \
                    $DATASETS_ARG \
                    $EXTRA_ARGS \
                    > $LOG 2>&1
            "
            sleep 1
        fi
        sessions+=("$session")
    done

    echo "  Aguardando onda $wave: ${sessions[*]}"
    while true; do
        ativos=0
        for s in "${sessions[@]}"; do
            tmux has-session -t "$s" 2>/dev/null && ativos=$((ativos + 1))
        done
        [ $ativos -eq 0 ] && break
        echo "  [$( date '+%H:%M' )] $ativos processo(s) ainda em execucao..."
        sleep 60
    done
    echo "  Onda $wave concluida."
done

echo ""
echo "Todos os $total experimentos concluidos (composition=$COMPOSITION)."
echo "Sessoes:    tmux ls 2>/dev/null | grep heteroB_"
echo "Resultados: $SCRIPT_DIR/results/neural/"
