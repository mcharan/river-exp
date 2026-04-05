#!/bin/bash
# ==============================================================================
# Execução em ondas — ARTESubspaceNN e ARTESoftResetNN
#
# Uso:
#   bash experiments/neural_arte/run_arte_member.sh --arch subspace
#   bash experiments/neural_arte/run_arte_member.sh --arch soft_reset
#   bash experiments/neural_arte/run_arte_member.sh --arch subspace --wave 3 --gpu 0
#   bash experiments/neural_arte/run_arte_member.sh --arch soft_reset --n_reset_layers 2
#   bash experiments/neural_arte/run_arte_member.sh --arch soft_reset --no_drift
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/experiments/neural_arte/run_arte_member.py"
LOGS_DIR="$SCRIPT_DIR/results/logs"

DATASETS_PATH="${DATASETS_PATH:-}"
DATASETS_PATH_REAL="${DATASETS_PATH_REAL:-}"

WAVE_SIZE=3
ARCH=""
GPU=0
N_MODELS=30
COMPOSITION="abc"
N_RESET_LAYERS=1
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)           ARCH="$2";           shift 2 ;;
        --wave)           WAVE_SIZE="$2";      shift 2 ;;
        --gpu)            GPU="$2";            shift 2 ;;
        --n_models)       N_MODELS="$2";       shift 2 ;;
        --composition)    COMPOSITION="$2";    shift 2 ;;
        --n_reset_layers) N_RESET_LAYERS="$2"; shift 2 ;;
        --no_drift)       EXTRA_ARGS="$EXTRA_ARGS --no_drift"; shift ;;
        *) echo "Argumento desconhecido: $1"; exit 1 ;;
    esac
done

if [ -z "$ARCH" ]; then
    echo "[ERRO] --arch é obrigatório. Use: --arch subspace  ou  --arch soft_reset"
    exit 1
fi

if [ "$ARCH" != "subspace" ] && [ "$ARCH" != "soft_reset" ]; then
    echo "[ERRO] --arch deve ser 'subspace' ou 'soft_reset', recebido: '$ARCH'"
    exit 1
fi

# Ordenados do mais rápido ao mais lento
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

# ------------------------------------------------------------------
# DATASETS_PATH validation
# ------------------------------------------------------------------
DATASETS_ARG=""
if [ -n "$DATASETS_PATH" ]; then
    if [ ! -d "$DATASETS_PATH" ]; then
        echo "[ERRO] DATASETS_PATH não existe: $DATASETS_PATH"
        exit 1
    fi
    if ! ls "$DATASETS_PATH"/*.arff &>/dev/null; then
        echo "[ERRO] Nenhum arquivo .arff encontrado em: $DATASETS_PATH"
        exit 1
    fi
    DATASETS_ARG="--datasets_path $DATASETS_PATH"
else
    echo "[AVISO] DATASETS_PATH não definido — usando o default do código."
    echo "        Para sobrescrever: export DATASETS_PATH=/caminho/datasets"
    echo ""
fi

# Fallback para datasets reais (quando DATASETS_PATH aponta para pasta mini)
if [ -n "$DATASETS_PATH_REAL" ]; then
    if [ ! -d "$DATASETS_PATH_REAL" ]; then
        echo "[ERRO] DATASETS_PATH_REAL não existe: $DATASETS_PATH_REAL"
        exit 1
    fi
    DATASETS_ARG="$DATASETS_ARG --datasets_path_real $DATASETS_PATH_REAL"
else
    echo "[AVISO] DATASETS_PATH_REAL não definido — datasets reais usarão o mesmo caminho."
    echo "        Para separar: export DATASETS_PATH_REAL=.../datasets"
    echo ""
fi

mkdir -p "$LOGS_DIR" "$SCRIPT_DIR/results/neural"

# ------------------------------------------------------------------
# Arch-specific args
# ------------------------------------------------------------------
ARCH_ARGS="--arch $ARCH --n_models $N_MODELS"
if [ "$ARCH" = "soft_reset" ]; then
    ARCH_ARGS="$ARCH_ARGS --composition $COMPOSITION --n_reset_layers $N_RESET_LAYERS"
fi

total=${#datasets[@]}
wave=0

echo "============================================================"
echo " arte_member — arch=$ARCH | ondas de $WAVE_SIZE | GPU=$GPU"
echo " n_models=$N_MODELS | $total datasets | Logs: $LOGS_DIR"
if [ "$ARCH" = "soft_reset" ]; then
    echo " composition=$COMPOSITION | n_reset_layers=$N_RESET_LAYERS"
fi
echo "============================================================"

i=0
while [ $i -lt $total ]; do
    wave=$((wave + 1))
    echo ""
    echo "--- Onda $wave ---"

    sessions=()
    for (( j=0; j<WAVE_SIZE && i<total; j++, i++ )); do
        ds="${datasets[$i]}"
        session="arteM_${ARCH}_${ds}"
        LOG="$LOGS_DIR/${session}.log"

        if screen -ls | grep -q "$session"; then
            echo "  [SKIP] $session já está ativa"
        else
            echo "  Disparando: $session"
            screen -dmS "$session" bash -c "
                cd $SCRIPT_DIR
                CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SCRIPT \
                    --dataset $ds \
                    --seed 123456789 \
                    --lambda_val 6 \
                    --window 500 \
                    $ARCH_ARGS \
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
            screen -ls | grep -q "$s" && ativos=$((ativos + 1))
        done
        [ $ativos -eq 0 ] && break
        echo "  [$( date '+%H:%M' )] $ativos processo(s) ainda em execução..."
        sleep 60
    done
    echo "  Onda $wave concluída."
done

echo ""
echo "Todos os $total experimentos concluídos (arch=$ARCH)."
echo "Sessões:    screen -ls | grep arteM_"
echo "Resultados: $SCRIPT_DIR/results/neural/"
