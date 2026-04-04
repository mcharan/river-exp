#!/bin/bash
# ==============================================================================
# Geração de Datasets Sintéticos via MOA CLI
#
# Uso:
#   bash generate_moa_datasets.sh [OPÇÕES]
#
# Opções:
#   --mode full|reduced   full = 1M instâncias, 3 drifts a cada 250k (padrão)
#                         reduced = 50k instâncias, 3 drifts a cada 12500
#   --output-dir <dir>    Diretório de saída dos ARFFs (padrão: diretório atual)
#   --moa-dir <dir>       Caminho para a pasta lib do MOA
#                         (padrão: /home/marcelo.charan1/Documents/moa/moa-bin/moa-release-2021.07.0/lib)
#
# Exemplos:
#   bash generate_moa_datasets.sh
#   bash generate_moa_datasets.sh --mode reduced --output-dir /tmp/datasets_mini
#   bash generate_moa_datasets.sh --mode full --output-dir /data/datasets
#
# Proporcionalidade do modo reduced (fator 1/20 em relação ao full):
#   Instâncias  : 1.000.000 → 50.000
#   Período drift: 250.000  → 12.500  (posição dos 3 cortes de drift)
#   Largura gradual: 50.000 → 2.500   (janela de transição suave)
#   Largura abrupta:     50 → 50      (já mínima; não escala para baixo)
#   Velocidade RBF:  mantém mesma deriva acumulada relativa ao tamanho do stream
#                    (s_reduced = s_full × 20, ex: 0.001 → 0.02)
# ==============================================================================

# ---- Defaults ---------------------------------------------------------------
MODE="full"
OUTPUT_DIR="."
BASE_DIR="/home/marcelo.charan1/Documents/moa/moa-bin/moa-release-2021.07.0/lib"

# ---- Parse de Argumentos ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --moa-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,28p' "$0"   # imprime o bloco de ajuda acima
            exit 0
            ;;
        *)
            echo "Argumento desconhecido: $1"
            echo "Use --help para ver as opções."
            exit 1
            ;;
    esac
done

# ---- Validação do modo ------------------------------------------------------
if [ "$MODE" != "full" ] && [ "$MODE" != "reduced" ]; then
    echo "Erro: --mode deve ser 'full' ou 'reduced' (recebido: '$MODE')"
    exit 1
fi

# ---- Parâmetros do stream conforme o modo -----------------------------------
if [ "$MODE" = "full" ]; then
    N_INSTANCES=1000000
    DRIFT_PERIOD=250000
    WIDTH_ABRUPT=50
    WIDTH_GRADUAL=50000
    RBF_FAST_SPEED=".001"
    RBF_MOD_SPEED=".0001"
else
    # reduced: fator de escala 1/20 (50k/1M)
    N_INSTANCES=50000
    DRIFT_PERIOD=12500
    WIDTH_ABRUPT=50        # já é mínimo; não reduzir mais
    WIDTH_GRADUAL=2500     # 50000 / 20
    # RBF: mantém a mesma velocidade do full — escalar por 20 tornaria
    # os conceitos inteiramente inestáveis em 50k instâncias (35% acc observado)
    RBF_FAST_SPEED=".001"
    RBF_MOD_SPEED=".0001"
fi

# ---- Configuração do MOA ----------------------------------------------------
MOA_JAR="${BASE_DIR}/moa.jar"
SIZEOFAG="${BASE_DIR}/sizeofag-1.0.4.jar"

if [ ! -f "$MOA_JAR" ]; then
    echo "Erro: moa.jar não encontrado em '${BASE_DIR}'."
    echo "Ajuste --moa-dir ou edite BASE_DIR no início do script."
    exit 1
fi

# Inclui o agente de memória somente se o jar existir
if [ -f "$SIZEOFAG" ]; then
    JAVA_CMD="java -cp $MOA_JAR -javaagent:$SIZEOFAG moa.DoTask"
else
    echo "Aviso: sizeofag não encontrado; executando sem -javaagent."
    JAVA_CMD="java -cp $MOA_JAR moa.DoTask"
fi

# ---- Cria diretório de saída ------------------------------------------------
mkdir -p "$OUTPUT_DIR"

echo "=== Iniciando Geração de Datasets via MOA CLI ==="
echo "Modo         : $MODE"
echo "Instâncias   : $N_INSTANCES"
echo "Período drift: $DRIFT_PERIOD  (3 drifts em $((DRIFT_PERIOD)), $((DRIFT_PERIOD*2)), $((DRIFT_PERIOD*3)))"
echo "Largura abr. : $WIDTH_ABRUPT  | Largura grad.: $WIDTH_GRADUAL"
echo "Saída        : $OUTPUT_DIR"
echo ""

# ---------------------------------------------------------
# 1. AGRAWAL (Abrupt vs Gradual)
# 4 conceitos encadeados: f=1 → f=2 → f=3(default) → f=4
# ---------------------------------------------------------

echo "Gerando agrawal_a.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/agrawal_a.arff -m $N_INSTANCES \
 -s (ConceptDriftStream \
      -s (generators.AgrawalGenerator -f 1) \
      -d (ConceptDriftStream \
           -s (generators.AgrawalGenerator -f 2) \
           -d (ConceptDriftStream \
                -s (generators.AgrawalGenerator) \
                -d (generators.AgrawalGenerator -f 4) \
                -w $WIDTH_ABRUPT -p $DRIFT_PERIOD) \
           -w $WIDTH_ABRUPT -p $DRIFT_PERIOD) \
      -w $WIDTH_ABRUPT -p $DRIFT_PERIOD)"

echo "Gerando agrawal_g.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/agrawal_g.arff -m $N_INSTANCES \
 -s (ConceptDriftStream \
      -s (generators.AgrawalGenerator -f 1) \
      -d (ConceptDriftStream \
           -s (generators.AgrawalGenerator -f 2) \
           -d (ConceptDriftStream \
                -s (generators.AgrawalGenerator) \
                -d (generators.AgrawalGenerator -f 4) \
                -w $WIDTH_GRADUAL -p $DRIFT_PERIOD) \
           -w $WIDTH_GRADUAL -p $DRIFT_PERIOD) \
      -w $WIDTH_GRADUAL -p $DRIFT_PERIOD)"

# ---------------------------------------------------------
# 2. LED (Abrupt vs Gradual)
# 4 conceitos com número crescente de atributos com drift: 1,3,5,7
# ---------------------------------------------------------

echo "Gerando led_a.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/led_a.arff -m $N_INSTANCES \
 -s (ConceptDriftStream \
      -s (generators.LEDGeneratorDrift -d 1) \
      -d (ConceptDriftStream \
           -s (generators.LEDGeneratorDrift -d 3) \
           -d (ConceptDriftStream \
                -s (generators.LEDGeneratorDrift -d 5) \
                -d (generators.LEDGeneratorDrift -d 7) \
                -w $WIDTH_ABRUPT -p $DRIFT_PERIOD) \
           -w $WIDTH_ABRUPT -p $DRIFT_PERIOD) \
      -w $WIDTH_ABRUPT -p $DRIFT_PERIOD)"

echo "Gerando led_g.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/led_g.arff -m $N_INSTANCES \
 -s (ConceptDriftStream \
      -s (generators.LEDGeneratorDrift -d 1) \
      -d (ConceptDriftStream \
           -s (generators.LEDGeneratorDrift -d 3) \
           -d (ConceptDriftStream \
                -s (generators.LEDGeneratorDrift -d 5) \
                -d (generators.LEDGeneratorDrift -d 7) \
                -w $WIDTH_GRADUAL -p $DRIFT_PERIOD) \
           -w $WIDTH_GRADUAL -p $DRIFT_PERIOD) \
      -w $WIDTH_GRADUAL -p $DRIFT_PERIOD)"

# ---------------------------------------------------------
# 3. SEA (Abrupt vs Gradual)
# 4 conceitos: f=1 → f=2 → f=3(default) → f=4
# ---------------------------------------------------------

echo "Gerando sea_a.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/sea_a.arff -m $N_INSTANCES \
 -s (ConceptDriftStream \
      -s (generators.SEAGenerator -f 1) \
      -d (ConceptDriftStream \
           -s (generators.SEAGenerator -f 2) \
           -d (ConceptDriftStream \
                -s (generators.SEAGenerator) \
                -d (generators.SEAGenerator -f 4) \
                -w $WIDTH_ABRUPT -p $DRIFT_PERIOD) \
           -w $WIDTH_ABRUPT -p $DRIFT_PERIOD) \
      -w $WIDTH_ABRUPT -p $DRIFT_PERIOD)"

echo "Gerando sea_g.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/sea_g.arff -m $N_INSTANCES \
 -s (ConceptDriftStream \
      -s (generators.SEAGenerator -f 1) \
      -d (ConceptDriftStream \
           -s (generators.SEAGenerator -f 2) \
           -d (ConceptDriftStream \
                -s (generators.SEAGenerator) \
                -d (generators.SEAGenerator -f 4) \
                -w $WIDTH_GRADUAL -p $DRIFT_PERIOD) \
           -w $WIDTH_GRADUAL -p $DRIFT_PERIOD) \
      -w $WIDTH_GRADUAL -p $DRIFT_PERIOD)"

# ---------------------------------------------------------
# 4. MIXED (Abrupt vs Gradual)
# MixedGenerator tem 2 conceitos (-f 0 e -f 1); alternamos 4 vezes.
# ---------------------------------------------------------

echo "Gerando mixed_a.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/mixed_a.arff -m $N_INSTANCES \
 -s (ConceptDriftStream \
      -s (generators.MixedGenerator -f 1) \
      -d (ConceptDriftStream \
           -s (generators.MixedGenerator -f 2) \
           -d (ConceptDriftStream \
                -s (generators.MixedGenerator -f 1) \
                -d (generators.MixedGenerator -f 2) \
                -w $WIDTH_ABRUPT -p $DRIFT_PERIOD) \
           -w $WIDTH_ABRUPT -p $DRIFT_PERIOD) \
      -w $WIDTH_ABRUPT -p $DRIFT_PERIOD)"

echo "Gerando mixed_g.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/mixed_g.arff -m $N_INSTANCES \
 -s (ConceptDriftStream \
      -s (generators.MixedGenerator -f 1) \
      -d (ConceptDriftStream \
           -s (generators.MixedGenerator -f 2) \
           -d (ConceptDriftStream \
                -s (generators.MixedGenerator -f 1) \
                -d (generators.MixedGenerator -f 2) \
                -w $WIDTH_GRADUAL -p $DRIFT_PERIOD) \
           -w $WIDTH_GRADUAL -p $DRIFT_PERIOD) \
      -w $WIDTH_GRADUAL -p $DRIFT_PERIOD)"

# ---------------------------------------------------------
# 5. RBF (Fast vs Moderate)
# Drift contínuo via RandomRBFGeneratorDrift (sem ConceptDriftStream).
# Velocidade (s) escalada proporcionalmente no modo reduced para manter
# o mesmo deslocamento acumulado relativo ao tamanho do stream.
# ---------------------------------------------------------

echo "Gerando rbf_f.arff (Fast, s=$RBF_FAST_SPEED)..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/rbf_f.arff -m $N_INSTANCES \
 -s (generators.RandomRBFGeneratorDrift -c 5 -s $RBF_FAST_SPEED)"

echo "Gerando rbf_m.arff (Moderate, s=$RBF_MOD_SPEED)..."
$JAVA_CMD "WriteStreamToARFFFile -f $OUTPUT_DIR/rbf_m.arff -m $N_INSTANCES \
 -s (generators.RandomRBFGeneratorDrift -c 5 -s $RBF_MOD_SPEED)"

echo ""
echo "=== Geração Concluída! ==="
echo "Arquivos gerados em: $(realpath "$OUTPUT_DIR")"
