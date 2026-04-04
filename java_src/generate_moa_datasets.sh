#!/bin/bash

BASE_DIR="/home/marcelo.charan1/Documents/moa/moa-bin/moa-release-2021.07.0/lib"
# Define o caminho para o seu moa.jar (ajuste se necessário)
MOA_JAR=${BASE_DIR}"/moa.jar"
# Se tiver o sizeofag.jar, inclua, senão pode remover o -javaagent
JAVA_CMD="java -cp $MOA_JAR -javaagent:${BASE_DIR}/sizeofag-1.0.4.jar moa.DoTask"

echo "=== Iniciando Geração de Datasets via MOA CLI ==="

# ---------------------------------------------------------
# 1. AGRAWAL (Abrupt vs Gradual)
# Nota: A diferença é o parâmetro -w (width). 50 para abrupto, 50000 para gradual.
# ---------------------------------------------------------

echo "Gerando agrawal_a.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f agrawal_a.arff -m 1000000 -s (ConceptDriftStream -s (generators.AgrawalGenerator -f 1) -d (ConceptDriftStream -s (generators.AgrawalGenerator -f 2) -d (ConceptDriftStream -s (generators.AgrawalGenerator) -d (generators.AgrawalGenerator -f 4) -w 50 -p 250000 ) -w 50 -p 250000 ) -w 50 -p 250000)"

echo "Gerando agrawal_g.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f agrawal_g.arff -m 1000000 -s (ConceptDriftStream -s (generators.AgrawalGenerator -f 1) -d (ConceptDriftStream -s (generators.AgrawalGenerator -f 2) -d (ConceptDriftStream -s (generators.AgrawalGenerator) -d (generators.AgrawalGenerator -f 4) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000)"

# ---------------------------------------------------------
# 2. LED (Abrupt vs Gradual)
# Configuração retirada do 
# ---------------------------------------------------------

echo "Gerando led_a.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f led_a.arff -m 1000000 -s (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 1) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 3) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 5) -d (generators.LEDGeneratorDrift -d 7) -w 50 -p 250000 ) -w 50 -p 250000 ) -w 50 -p 250000)"

echo "Gerando led_g.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f led_g.arff -m 1000000 -s (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 1) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 3) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 5) -d (generators.LEDGeneratorDrift -d 7) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000)"

# ---------------------------------------------------------
# 3. SEA (Abrupt vs Gradual)
# Configuração retirada do 
# ---------------------------------------------------------

echo "Gerando sea_a.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f sea_a.arff -m 1000000 -s (ConceptDriftStream -s (generators.SEAGenerator -f 1) -d (ConceptDriftStream -s (generators.SEAGenerator -f 2) -d (ConceptDriftStream -s (generators.SEAGenerator) -d (generators.SEAGenerator -f 4) -w 50 -p 250000 ) -w 50 -p 250000 ) -w 50 -p 250000)"

echo "Gerando sea_g.arff..."
$JAVA_CMD "WriteStreamToARFFFile -f sea_g.arff -m 1000000 -s (ConceptDriftStream -s (generators.SEAGenerator -f 1) -d (ConceptDriftStream -s (generators.SEAGenerator -f 2) -d (ConceptDriftStream -s (generators.SEAGenerator) -d (generators.SEAGenerator -f 4) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000)"

# ---------------------------------------------------------
# 4. RBF (Fast vs Moderate)
# Configuração retirada do 
# Nota: Aqui ele usa o RandomRBFGeneratorDrift direto, sem ConceptDriftStream aninhado
# ---------------------------------------------------------

echo "Gerando rbf_f.arff (Fast)..."
$JAVA_CMD "WriteStreamToARFFFile -f rbf_f.arff -m 1000000 -s (generators.RandomRBFGeneratorDrift -c 5 -s .001)"

echo "Gerando rbf_m.arff (Moderate)..."
$JAVA_CMD "WriteStreamToARFFFile -f rbf_m.arff -m 1000000 -s (generators.RandomRBFGeneratorDrift -c 5 -s .0001)"

echo "=== Geração Concluída! ==="
