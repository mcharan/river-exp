#!/bin/bash
# ==============================================================================
# ABLATION MINI — datasets reduzidos (50k inst), 2 eixos de investigação
#
# Eixo 1 — Composição (n=30 fixo):
#   GPU 0: abc n=30  |  current n=30
#   GPU 1: abc_proj n=30
#
# Eixo 2 — n_models (composition=abc):
#   GPU 0: abc n=10  |  abc n=20
#   GPU 1: abc n=50
#
# Uso:
#   bash experiments/neural_arte/run_ablation_mini.sh              # ambos os eixos
#   bash experiments/neural_arte/run_ablation_mini.sh --eixo comp  # só composição
#   bash experiments/neural_arte/run_ablation_mini.sh --eixo nmod  # só n_models
#   bash experiments/neural_arte/run_ablation_mini.sh --wave 4     # ondas de 4
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/src/neural_arte/neural_arte.py"

SEED=123456789
LAMBDA=6
WINDOW=500
DATASETS_PATH="${DATASETS_PATH:-}"
LOGS_DIR="$SCRIPT_DIR/results/logs"
WAVE_SIZE=4
EIXO="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --wave)  WAVE_SIZE="$2"; shift 2 ;;
        --eixo)  EIXO="$2";      shift 2 ;;
        *) echo "Argumento desconhecido: $1"; exit 1 ;;
    esac
done

# Datasets mini — sintéticos (drift explícito) + reais pequenos
# Ordenados do mais rápido ao mais lento
ALL_DATASETS=(
    "keystroke" "ozone" "outdoor" "gassensor"
    "electricity" "shuttle" "rialto" "gmsc"
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
    echo "        Para datasets mini: export DATASETS_PATH=.../datasets/mini"
    echo ""
fi

mkdir -p "$LOGS_DIR" "$SCRIPT_DIR/results/neural"

# ------------------------------------------------------------------------------
# Função: roda uma variante em ondas numa GPU (bloqueante)
# run_variant <tag> <gpu> <n_models> <composition>
# ------------------------------------------------------------------------------
run_variant() {
    local tag="$1" gpu="$2" n_models="$3" composition="$4"
    local total=${#ALL_DATASETS[@]}
    local wave=0 i=0

    echo "  [GPU $gpu] $tag (comp=$composition, n=$n_models)"

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
# Eixo composição — n=30 fixo
# GPU 0: abc + current  (sequencial, mesma GPU)
# GPU 1: abc_proj       (paralelo com GPU 0)
# ------------------------------------------------------------------------------
run_eixo_composicao() {
    echo "============================================================"
    echo " EIXO COMPOSIÇÃO (n=30) | ondas de $WAVE_SIZE"
    echo " GPU 0: abc → current  |  GPU 1: abc_proj"
    echo "============================================================"

    {
        run_variant "mini_abc30"     0 30 "abc"
        run_variant "mini_current30" 0 30 "current"
    } &

    run_variant "mini_abcproj30" 1 30 "abc_proj" &

    wait
    echo "Eixo composição concluído."
}

# ------------------------------------------------------------------------------
# Eixo n_models — composition=abc
# GPU 0: n=10 → n=20   (sequencial, mesma GPU)
# GPU 1: n=50           (paralelo com GPU 0)
# ------------------------------------------------------------------------------
run_eixo_nmodels() {
    echo "============================================================"
    echo " EIXO N_MODELS (composition=abc) | ondas de $WAVE_SIZE"
    echo " GPU 0: n=10 → n=20  |  GPU 1: n=50"
    echo "============================================================"

    {
        run_variant "mini_abc10" 0 10 "abc"
        run_variant "mini_abc20" 0 20 "abc"
    } &

    run_variant "mini_abc50" 1 50 "abc" &

    wait
    echo "Eixo n_models concluído."
}

# ------------------------------------------------------------------------------
# Execução
# ------------------------------------------------------------------------------
if [ "$EIXO" = "comp" ]; then
    run_eixo_composicao
elif [ "$EIXO" = "nmod" ]; then
    run_eixo_nmodels
else
    run_eixo_composicao
    echo ""
    run_eixo_nmodels
fi

echo ""
echo "Ablação mini concluída."
echo "Logs:       $LOGS_DIR/mini_<variante>_<dataset>.log"
echo "Resultados: $SCRIPT_DIR/results/neural/"
