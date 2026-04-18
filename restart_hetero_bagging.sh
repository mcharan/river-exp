#!/bin/bash
# Restart HeteroBagging experiments with fixed CUDA_VISIBLE_DEVICES handling
# Run this script after applying the fix to run_hetero_bagging.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/results/logs"
PYTHON="nice -n 10 /home/marcelo/.venv/river-exp/bin/python3"
DP=/home/marcelo/Documents/code/datasets/mini
DPR=/home/marcelo/Documents/code/datasets

echo "=== Stopping old abc/abc_proj orchestrators ==="

# Kill orchestrator bash processes (not heterogeneous ones)
for pid in $(pgrep -f "run_hetero_bagging.sh.*--composition abc" 2>/dev/null); do
    proc=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$proc" | grep -qv "heterogeneous"; then
        kill "$pid" 2>/dev/null && echo "Killed orchestrator PID $pid"
    fi
done

sleep 2

echo "=== Stopping stuck hetero_bagging CPU workers ==="
pkill -f "hetero_bagging.py.*--gpu -1" 2>/dev/null
sleep 2

echo "=== Cleaning tmux sessions (old wave naming without DRIFT_TAG) ==="
for ds in keystroke ozone outdoor gassensor electricity shuttle rialto gmsc covtype airlines sea_a sea_g led_a led_g agrawal_a agrawal_g mixed_a mixed_g rbf_f rbf_m; do
    tmux kill-session -t "heteroB_abc_${ds}" 2>/dev/null
    tmux kill-session -t "heteroB_abc_proj_${ds}" 2>/dev/null
done

echo "=== Cleaning empty log files ==="
find "$LOGS_DIR" -name "heteroB_abc_*.log" -empty -delete 2>/dev/null
find "$LOGS_DIR" -name "heteroB_abc_proj_*.log" -empty -delete 2>/dev/null
echo "Empty logs removed."

echo "=== Relaunching orchestrators with fixed script ==="

# abc adwin (no --no_drift)
tmux new-session -d -s "hb_abc_adwin" bash -c "
    export PYTHON='$PYTHON'
    export DATASETS_PATH=$DP
    export DATASETS_PATH_REAL=$DPR
    export CUDA_VISIBLE_DEVICES=''
    cd $SCRIPT_DIR
    bash experiments/neural_arte/run_hetero_bagging.sh \
        --composition abc --wave 3 --gpu -1 \
        > $LOGS_DIR/orchestrator_hb_abc_adwin.log 2>&1
" && echo "Launched hb_abc_adwin"

sleep 2

# abc nodrift
tmux new-session -d -s "hb_abc_nodrift" bash -c "
    export PYTHON='$PYTHON'
    export DATASETS_PATH=$DP
    export DATASETS_PATH_REAL=$DPR
    export CUDA_VISIBLE_DEVICES=''
    cd $SCRIPT_DIR
    bash experiments/neural_arte/run_hetero_bagging.sh \
        --composition abc --wave 3 --gpu -1 --no_drift \
        > $LOGS_DIR/orchestrator_hb_abc_nodrift.log 2>&1
" && echo "Launched hb_abc_nodrift"

sleep 2

# abc_proj adwin
tmux new-session -d -s "hb_abc_proj_adwin" bash -c "
    export PYTHON='$PYTHON'
    export DATASETS_PATH=$DP
    export DATASETS_PATH_REAL=$DPR
    export CUDA_VISIBLE_DEVICES=''
    cd $SCRIPT_DIR
    bash experiments/neural_arte/run_hetero_bagging.sh \
        --composition abc_proj --wave 3 --gpu -1 \
        > $LOGS_DIR/orchestrator_hb_abc_proj_adwin.log 2>&1
" && echo "Launched hb_abc_proj_adwin"

sleep 2

# abc_proj nodrift
tmux new-session -d -s "hb_abc_proj_nodrift" bash -c "
    export PYTHON='$PYTHON'
    export DATASETS_PATH=$DP
    export DATASETS_PATH_REAL=$DPR
    export CUDA_VISIBLE_DEVICES=''
    cd $SCRIPT_DIR
    bash experiments/neural_arte/run_hetero_bagging.sh \
        --composition abc_proj --wave 3 --gpu -1 --no_drift \
        > $LOGS_DIR/orchestrator_hb_abc_proj_nodrift.log 2>&1
" && echo "Launched hb_abc_proj_nodrift"

echo ""
echo "=== Done. Check progress with: ==="
echo "  tail -5 $LOGS_DIR/orchestrator_hb_abc_adwin.log"
echo "  ls results/neural/HeteroBagging_*.csv 2>/dev/null | wc -l"
