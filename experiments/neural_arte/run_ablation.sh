#!/bin/bash
# ==============================================================================
# ABLATION: tamanho do ensemble (n_models) + composição abc_extended
#
# Fase 1: GPU 0 → abc n=10      |  GPU 1 → abc_extended n=30
# Fase 2: GPU 0 → abc n=60      |  GPU 1 → abc n=100
#
# Dentro de cada GPU, os datasets rodam em ondas de --wave N.
#
# Uso:
#   bash run_ablation.sh 1              # só fase 1
#   bash run_ablation.sh 2              # só fase 2
#   bash run_ablation.sh auto           # fase 1 → aguarda → fase 2
#   bash run_ablation.sh auto --wave 3  # idem, ondas de 3
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/src/neural_arte/neural_arte.py"

SEED=123456789
LAMBDA=6
WINDOW=500
DATASETS_PATH="${DATASETS_PATH:-}"
LOGS_DIR="$SCRIPT_DIR/results/logs"
WAVE_SIZE=3

FASE=${1:-1}
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --wave) WAVE_SIZE="$2"; shift 2 ;;
        *) echo "Argumento desconhecido: $1"; exit 1 ;;
    esac
done

# Ordenados do mais rápido ao mais lento
ALL_DATASETS=(
    "keystroke" "ozone" "outdoor" "gassensor"
    "electricity" "shuttle" "rialto" "gmsc" "covtype" "airlines"
    "sea_a" "sea_g" "led_a" "led_g"
    "agrawal_a" "agrawal_g" "mixed_a" "mixed_g" "rbf_f" "rbf_m"
)

DATASETS_ARG=""
if [ -n "$DATASETS_PATH" ]; then
    if [ ! -d "$DATASETS_PATH" ]; then
        echo "[ERRO] DATASETS_PATH não existe: $DATASETS_PATH"; exit 1
    fi
    if ! ls "$DATASETS_PATH"/*.arff &>/dev/null; then
        echo "[ERRO] Nenhum .arff encontrado em: $DATASETS_PATH"; exit 1
    fi
    DATASETS_ARG="--datasets_path $DATASETS_PATH"
else
    echo "[AVISO] DATASETS_PATH não definido — usando o default do código."
    echo ""
fi

mkdir -p "$LOGS_DIR" "$SCRIPT_DIR/results/neural"

# ------------------------------------------------------------------------------
# Função: roda uma variante em ondas numa GPU específica (bloqueante)
# run_variant <tag> <gpu> <n_models> <composition>
# ------------------------------------------------------------------------------
run_variant() {
    local tag="$1" gpu="$2" n_models="$3" composition="$4"
    local total=${#ALL_DATASETS[@]}
    local wave=0 i=0

    echo "  [GPU $gpu] $tag (n=$n_models, comp=$composition) — ondas de $WAVE_SIZE"

    while [ $i -lt $total ]; do
        wave=$((wave + 1))
        local sessions=()

        for (( j=0; j<WAVE_SIZE && i<total; j++, i++ )); do
            local ds="${ALL_DATASETS[$i]}"
            local session="${tag}_${ds}"
            local LOG="$LOGS_DIR/${session}.log"

            if screen -ls | grep -q "$session"; then
                echo "    [SKIP] $session já está ativa"
            else
                screen -dmS "$session" bash -c "
                    cd $SCRIPT_DIR
                    CUDA_VISIBLE_DEVICES=$gpu $PYTHON $SCRIPT \
                        --dataset $ds --seed $SEED \
                        --n_models $n_models \
                        --lambda_val $LAMBDA --window $WINDOW \
                        --composition $composition \
                        $DATASETS_ARG \
                        > $LOG 2>&1
                "
                sleep 1
            fi
            sessions+=("$session")
        done

        echo "    Onda $wave: aguardando ${sessions[*]}"
        while true; do
            local ativos=0
            for s in "${sessions[@]}"; do
                screen -ls | grep -q "$s" && ativos=$((ativos + 1))
            done
            [ $ativos -eq 0 ] && break
            echo "    [$( date '+%H:%M' )] $ativos processo(s) ativos..."
            sleep 60
        done
    done

    echo "  [GPU $gpu] $tag concluído."
}

# ------------------------------------------------------------------------------
# Fases
# ------------------------------------------------------------------------------
if [ "$FASE" = "1" ]; then
    echo "============================================================"
    echo " ABLATION — Fase 1 | ondas de $WAVE_SIZE"
    echo " GPU 0: abc n=10  |  GPU 1: abc_extended n=30"
    echo "============================================================"
    run_variant "abl_abc10"   0  10 "abc"          &
    run_variant "abl_abcext"  1  30 "abc_extended" &
    wait
    echo ""
    echo "Fase 1 concluída."

elif [ "$FASE" = "2" ]; then
    echo "============================================================"
    echo " ABLATION — Fase 2 | ondas de $WAVE_SIZE"
    echo " GPU 0: abc n=60  |  GPU 1: abc n=100"
    echo "============================================================"
    run_variant "abl_abc60"   0  60 "abc" &
    run_variant "abl_abc100"  1 100 "abc" &
    wait
    echo ""
    echo "Fase 2 concluída."

elif [ "$FASE" = "auto" ]; then
    echo "============================================================"
    echo " ABLATION — Auto (fase 1 → fase 2) | ondas de $WAVE_SIZE"
    echo "============================================================"
    bash "$0" 1 --wave "$WAVE_SIZE"
    echo ""
    echo "Iniciando fase 2..."
    bash "$0" 2 --wave "$WAVE_SIZE"
    echo ""
    echo "Ablação completa."

else
    echo "Uso: bash run_ablation.sh 1|2|auto [--wave N]"
    exit 1
fi

echo ""
echo "Logs:      tail -f $LOGS_DIR/abl_<variante>_<dataset>.log"
echo "Sessões:   screen -ls | grep abl_"
echo "Resultados: $SCRIPT_DIR/results/neural/"
