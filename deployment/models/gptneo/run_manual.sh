#!/bin/bash
# Minimal script to run resnet50 training on manual nodes
# Usage: ./run_manual.sh

set -e

# Configuration
CONFIG_PATH="configs/resnet50.yaml"
REPO_PATH="\$HOME/distqat"
INITIAL_PEERS_PATH="\$HOME/distqat/deployment/models/resnet50/initial_peers.txt"

# Monitor node (runs start_trainer_client.py)
MONITOR_HOST="ubuntu@204.12.163.10"  # Fill in monitor IP, e.g., "ubuntu@1.2.3.4"

# Worker nodes: hostname user@ip stage_index diloco_inner_steps
# Format: "user@ip:stage_index:inner_steps"
declare -a WORKERS=(
    # RTX A6000 nodes (430 inner steps)
    "ubuntu@64.247.196.66:1:430"
    "ubuntu@64.247.196.72:2:430"
    "ubuntu@64.247.196.96:3:430"
    "ubuntu@64.247.196.97:4:430"
    # A6000 nodes (500 inner steps)
    "ubuntu@64.247.196.18:1:500"
    "ubuntu@64.247.196.21:2:500"
    "ubuntu@64.247.196.23:3:500"
    "ubuntu@64.247.196.24:4:500"
    # A100 nodes (450 inner steps)
    "ubuntu@154.54.100.100:1:450"
    "ubuntu@154.54.100.109:2:450"
    "ubuntu@154.54.100.119:3:450"
    "ubuntu@154.54.100.122:4:450"
    # L40S nodes (320 inner steps)
    "ubuntu@216.81.248.71:1:320"
    "ubuntu@216.81.248.83:2:320"
    "ubuntu@216.81.248.64:3:320"
    "ubuntu@216.81.248.89:4:320"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# SSH options for non-interactive use
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

start_monitor() {
    if [ -z "$MONITOR_HOST" ]; then
        log_error "MONITOR_HOST not set. Please fill in the monitor IP."
        exit 1
    fi
    
    log_info "Starting monitor on $MONITOR_HOST..."
    ssh $SSH_OPTS "$MONITOR_HOST" "cd $REPO_PATH && \
        source .venv/bin/activate && \
        nohup python start_trainer_client.py --config-path $CONFIG_PATH \
        > logs/resnet50/trainer_client.log 2>&1 &"
    
    log_info "Monitor started. Waiting for initial peers file..."
    
    # Wait for initial peers file to be created
    for i in {1..60}; do
        if ssh $SSH_OPTS "$MONITOR_HOST" "test -s $INITIAL_PEERS_PATH" 2>/dev/null; then
            log_info "Initial peers file ready!"
            return 0
        fi
        echo -n "."
        sleep 3
    done
    
    log_error "Timeout waiting for initial peers file"
    exit 1
}

fetch_initial_peers() {
    log_info "Fetching initial peers from monitor..."
    INITIAL_PEERS=$(ssh $SSH_OPTS "$MONITOR_HOST" "cat $INITIAL_PEERS_PATH")
    log_info "Initial peers: $INITIAL_PEERS"
}

start_worker() {
    local worker_spec="$1"
    
    # Parse worker spec: user@ip:stage_index:inner_steps
    IFS=':' read -r host stage_idx inner_steps <<< "$worker_spec"
    
    log_info "Starting worker on $host (stage=$stage_idx, inner_steps=$inner_steps)..."
    
    ssh $SSH_OPTS "$host" "cd $REPO_PATH && \
        source .venv/bin/activate && \
        nohup python start_servers.py \
            --config-path $CONFIG_PATH \
            --initial-peers-path $INITIAL_PEERS_PATH \
            --num-servers 1 \
            --expert-index 0 \
            --stage-index $stage_idx \
            --diloco-inner-steps $inner_steps \
        > logs/resnet50/server_stage${stage_idx}.log 2>&1 &"
    
    log_info "Worker started on $host"
}

copy_initial_peers_to_workers() {
    log_info "Copying initial peers file to all workers..."
    
    for worker_spec in "${WORKERS[@]}"; do
        IFS=':' read -r host _ _ <<< "$worker_spec"
        log_info "  -> $host"
        
        # Ensure log directory exists and copy file
        ssh $SSH_OPTS "$host" "mkdir -p $REPO_PATH/logs/resnet50"
        ssh $SSH_OPTS "$MONITOR_HOST" "cat $INITIAL_PEERS_PATH" | \
            ssh $SSH_OPTS "$host" "cat > $INITIAL_PEERS_PATH"
    done
}

start_all_workers() {
    log_info "Starting all workers..."
    
    for worker_spec in "${WORKERS[@]}"; do
        start_worker "$worker_spec"
        sleep 2  # Small delay between starting workers
    done
}

stop_all() {
    log_info "Stopping all processes..."
    
    if [ -n "$MONITOR_HOST" ]; then
        log_info "Stopping monitor..."
        ssh $SSH_OPTS "$MONITOR_HOST" "pkill -f 'start_trainer_client.py' || true" 2>/dev/null || true
    fi
    
    for worker_spec in "${WORKERS[@]}"; do
        IFS=':' read -r host _ _ <<< "$worker_spec"
        log_info "Stopping worker on $host..."
        ssh $SSH_OPTS "$host" "pkill -f 'start_servers.py' || true" 2>/dev/null || true
    done
    
    log_info "All processes stopped"
}

show_status() {
    log_info "Checking process status..."
    
    if [ -n "$MONITOR_HOST" ]; then
        echo -e "\n${YELLOW}Monitor ($MONITOR_HOST):${NC}"
        ssh $SSH_OPTS "$MONITOR_HOST" "pgrep -fa 'start_trainer_client.py' || echo 'Not running'" 2>/dev/null || echo "Cannot connect"
    fi
    
    for worker_spec in "${WORKERS[@]}"; do
        IFS=':' read -r host stage_idx _ <<< "$worker_spec"
        echo -e "\n${YELLOW}Worker $host (stage $stage_idx):${NC}"
        ssh $SSH_OPTS "$host" "pgrep -fa 'start_servers.py' || echo 'Not running'" 2>/dev/null || echo "Cannot connect"
    done
}

show_logs() {
    local host="$1"
    local logfile="$2"
    
    if [ -z "$host" ]; then
        log_error "Usage: $0 logs <host> [logfile]"
        exit 1
    fi
    
    logfile="${logfile:-trainer_client.log}"
    
    log_info "Showing logs from $host ($logfile)..."
    ssh $SSH_OPTS "$host" "tail -f $REPO_PATH/logs/resnet50/$logfile"
}

usage() {
    echo "Usage: $0 {start|stop|status|logs|monitor|workers}"
    echo ""
    echo "Commands:"
    echo "  start     - Start monitor, wait for peers, then start all workers"
    echo "  stop      - Stop all processes on all nodes"
    echo "  status    - Show running processes on all nodes"
    echo "  logs      - Show logs: $0 logs <host> [logfile]"
    echo "  monitor   - Start only the monitor"
    echo "  workers   - Start only the workers (assumes initial_peers exists)"
    echo ""
    echo "Before running, edit this script to set:"
    echo "  - MONITOR_HOST"
    echo "  - WORKERS array with your node IPs"
}

# Main
case "${1:-}" in
    start)
        start_monitor
        fetch_initial_peers
        copy_initial_peers_to_workers
        start_all_workers
        log_info "All nodes started!"
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2" "$3"
        ;;
    monitor)
        start_monitor
        ;;
    workers)
        fetch_initial_peers
        copy_initial_peers_to_workers
        start_all_workers
        ;;
    *)
        usage
        exit 1
        ;;
esac
