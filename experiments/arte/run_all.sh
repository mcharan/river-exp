#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT="$SCRIPT_DIR/experiments/arte/run_experiments.py"

# Lista de Datasets da Tabela 14
datasets=(
    "airlines"
    "electricity"
    "gassensor"
    "gmsc"
    "keystroke"
    "outdoor"
    "ozone"
    "rialto"
    "shuttle"
    "covtype"
    "agrawal_a"
    "agrawal_g"
    "led_a"
    "led_g"
    "sea_a"
    "sea_g"
    "mixed_a"
    "rbf_f"
    "rbf_m"
)

echo "Disparando experimentos em paralelo..."

for ds in "${datasets[@]}"; do
    echo "Iniciando screen para: $ds"
    # Cria uma screen com o nome do dataset e executa o python
    screen -dmS "exp_$ds" bash -c "cd $SCRIPT_DIR && $PYTHON $SCRIPT --dataset $ds"

    # Pequena pausa para nao sobrecarregar o I/O de disco na largada
    sleep 2
done

echo "Todos os experimentos foram disparados!"
echo "Use 'screen -ls' para ver as sessoes."
echo "Use 'screen -r exp_nome_dataset' para ver o log de um especifico."


