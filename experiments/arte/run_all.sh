#!/bin/bash
# ==============================================================================
# Execução em ondas — evita sobrecarregar o servidor com muitos processos
#
# Uso:
#   bash experiments/arte/run_all.sh                  # onda de 4 (padrão)
#   bash experiments/arte/run_all.sh --wave 3         # onda de 3
#   bash experiments/arte/run_all.sh --wave 6         # onda de 6
#   bash experiments/arte/run_all.sh --mw 10          # min_window_length=10
#
# Datasets ordenados do menor para o maior (ondas terminam mais uniformemente)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/experiments/arte/run_experiments.py"
LOGS_DIR="$SCRIPT_DIR/results/logs"
DATASETS_PATH="${DATASETS_PATH:-}"

WAVE_SIZE=4
MW=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --wave) WAVE_SIZE="$2"; shift 2 ;;
        --mw)   MW="$2";        shift 2 ;;
        *) echo "Argumento desconhecido: $1"; exit 1 ;;
    esac
done

# Ordenados aproximadamente do mais rápido ao mais lento
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
    DATASETS_ARG="--datasets_path $DATASETS_PATH"
fi

mkdir -p "$LOGS_DIR"

total=${#datasets[@]}
wave=0

echo "============================================================"
echo " ARTE — execução em ondas de $WAVE_SIZE | mw=$MW"
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
        session="exp_${ds}_mw${MW}"
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
                    --n_models 100 \
                    --window_size 500 \
                    --adwin_min_window $MW \
                    $DATASETS_ARG \
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
echo "Resultados: $SCRIPT_DIR/results/arte/"


