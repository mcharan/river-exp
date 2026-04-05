#!/bin/bash
# ==============================================================================
# Execução em ondas — NeuralARTE
#
# Uso:
#   bash experiments/neural_arte/run_all.sh                    # onda de 4 (padrão)
#   bash experiments/neural_arte/run_all.sh --wave 3           # onda de 3
#   bash experiments/neural_arte/run_all.sh --composition abc  # composição do ensemble
#   bash experiments/neural_arte/run_all.sh --no_drift         # sem detector de drift
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/src/neural_arte/neural_arte.py"
LOGS_DIR="$SCRIPT_DIR/results/logs"
DATASETS_PATH="${DATASETS_PATH:-}"

WAVE_SIZE=4
COMPOSITION="abc"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --wave)        WAVE_SIZE="$2";   shift 2 ;;
        --composition) COMPOSITION="$2"; shift 2 ;;
        --no_drift)    EXTRA_ARGS="$EXTRA_ARGS --no_drift"; shift ;;
        *) echo "Argumento desconhecido: $1"; exit 1 ;;
    esac
done

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
    "agrawal_a"
    "agrawal_g"
    "led_a"
    "led_g"
    "sea_a"
    "sea_g"
    "mixed_a"
    "mixed_g"
    "rbf_f"
    "rbf_m"
)

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

mkdir -p "$LOGS_DIR"

total=${#datasets[@]}
wave=0

echo "============================================================"
echo " NeuralARTE — ondas de $WAVE_SIZE | composition=$COMPOSITION"
echo " $total datasets | Logs: $LOGS_DIR"
echo "============================================================"

i=0
while [ $i -lt $total ]; do
    wave=$((wave + 1))
    echo ""
    echo "--- Onda $wave ---"

    sessions=()
    for (( j=0; j<WAVE_SIZE && i<total; j++, i++ )); do
        ds="${datasets[$i]}"
        session="neural_${ds}_${COMPOSITION}"
        LOG="$LOGS_DIR/${session}.log"

        if screen -ls | grep -q "$session"; then
            echo "  [SKIP] $session já está ativa"
        else
            echo "  Disparando: $session"
            screen -dmS "$session" bash -c "
                cd $SCRIPT_DIR
                $PYTHON $SCRIPT \
                    --dataset $ds \
                    --seed 123456789 \
                    --n_models 30 \
                    --composition $COMPOSITION \
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
echo "Todos os $total experimentos concluídos."
echo "Resultados: $SCRIPT_DIR/results/neural/"


