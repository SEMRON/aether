#!/bin/bash
# Minimal script to run resnet50 training on manual nodes
# Usage: ./run_manual.sh
#
# NOTE: Start the monitor manually first:
#   ssh ubuntu@69.67.150.129
#   cd ~/distqat && source .venv/bin/activate
#   python start_trainer_client.py --config-path configs/resnet50.yaml

set -e

# Configuration
CONFIG_PATH="configs/resnet50.yaml"
REPO_PATH="\$HOME/distqat"
# INITIAL_PEERS_PATH="\$HOME/distqat/deployment/models/resnet50/initial_peers.txt"
INITIAL_PEER="/ip4/69.67.150.129/tcp/50000/p2p/QmYLhzANGdRNQjLx7wu3EifRfJbowCXjkfCSqEmrmSSCJ4"

# Monitor host (for fetching initial_peers.txt)
MONITOR_HOST="ubuntu@69.67.150.129"
WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY environment variable is required}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable is required}"

# Worker nodes: hostname user@ip stage_index diloco_inner_steps
# Format: "user@ip:expert_index:inner_steps"
declare -a WORKERS=(
    # RTX6000 Ada nodes (430 inner steps)
    "ubuntu@216.81.248.170:1:430"   # resnet50-test1_1
    "ubuntu@64.247.196.66:2:430"    # resnet50-test1_2
    "ubuntu@64.247.196.72:3:430"    # resnet50-test1_3
    "ubuntu@64.247.196.96:4:430"    # resnet50-test1_4
    # A6000 nodes (500 inner steps)
    "ubuntu@64.247.206.226:7:500"   # resnet50-test2_1
    "ubuntu@64.247.196.18:8:500"    # resnet50-test2_2
    "ubuntu@64.247.196.21:9:500"    # resnet50-test2_3
    "ubuntu@64.247.196.23:10:500"   # resnet50-test2_4
    # A100 nodes (450 inner steps)
    "ubuntu@154.54.100.100:11:450"  # resnet50-test3_1
    "ubuntu@154.54.100.109:12:450"  # resnet50-test3_2
    "ubuntu@154.54.100.110:13:450"  # resnet50-test3_3
    "ubuntu@154.54.100.119:14:450"  # resnet50-test3_4
    # L40S nodes (320 inner steps)
    "ubuntu@216.81.245.209:15:320"  # resnet50-test4_1
    "ubuntu@216.81.248.71:16:320"   # resnet50-test4_2
    "ubuntu@216.81.248.83:17:320"   # resnet50-test4_3
    "ubuntu@216.81.248.89:18:320"   # resnet50-test4_4
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

fetch_initial_peers() {
    log_info "Fetching initial peers from monitor..."
    INITIAL_PEERS=$(ssh $SSH_OPTS "$MONITOR_HOST" "cat $INITIAL_PEERS_PATH")
    log_info "Initial peers: $INITIAL_PEERS"
}

start_worker() {
    local worker_spec="$1"
    
    # Parse worker spec: user@ip:expert_index:inner_steps
    IFS=':' read -r host expert_idx inner_steps <<< "$worker_spec"
    
    log_info "Starting worker on $host (expert=$expert_idx, inner_steps=$inner_steps)..."
    
    ssh $SSH_OPTS "$host" bash -c "'
        cd $REPO_PATH && \
        source .venv/bin/activate && \
        export WANDB_API_KEY=$WANDB_API_KEY && \
        export HF_TOKEN=$HF_TOKEN && \
        nohup python start_servers.py \
            --config-path $CONFIG_PATH \
            --network-initial-peers $INITIAL_PEER \
            --num-servers 1 \
            --expert-index $expert_idx \
            --diloco-inner-steps $inner_steps \
        > logs/resnet50/server_${expert_idx}.log 2>&1 & 
    '"
    
    log_info "Worker started on $host"
}

copy_initial_peers_to_workers() {
    log_info "Copying initial peers file to all workers..."
    
    for worker_spec in "${WORKERS[@]}"; do
        IFS=':' read -r host _ _ <<< "$worker_spec"
        log_info "  -> $host"
        
        # Ensure log directory exists and copy file
        ssh $SSH_OPTS "$host" "mkdir -p $REPO_PATH/logs/resnet50"
        ssh $SSH_OPTS "$host" "echo $INITIAL_PEER > $INITIAL_PEERS_PATH"
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
    log_info "Stopping all worker processes..."
    
    for worker_spec in "${WORKERS[@]}"; do
        IFS=':' read -r host _ _ <<< "$worker_spec"
        log_info "Stopping worker on $host..."
        ssh $SSH_OPTS "$host" "pkill -f 'start_servers.py' || true" 2>/dev/null || true
    done
    
    log_info "All workers stopped"
}

show_status() {
    log_info "Checking process status..."
    
    for worker_spec in "${WORKERS[@]}"; do
        IFS=':' read -r host expert_idx _ <<< "$worker_spec"
        echo -e "\n${YELLOW}Worker $host (expert $expert_idx):${NC}"
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
    
    logfile="${logfile:-server.log}"
    
    log_info "Showing logs from $host ($logfile)..."
    ssh $SSH_OPTS "$host" "tail -f $REPO_PATH/logs/resnet50/$logfile"
}

usage() {
    echo "Usage: $0 {start|stop|status|logs}"
    echo ""
    echo "Commands:"
    echo "  start     - Copy initial peers and start all workers"
    echo "  stop      - Stop all worker processes"
    echo "  status    - Show running processes on all nodes"
    echo "  logs      - Show logs: $0 logs <host> [logfile]"
    echo ""
    echo "NOTE: Start the monitor manually first:"
    echo "  ssh $MONITOR_HOST"
    echo "  cd ~/distqat && source .venv/bin/activate"
    echo "  python start_trainer_client.py --config-path $CONFIG_PATH"
}

# Main
case "${1:-}" in
    start)
        start_all_workers
        log_info "All workers started!"
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
    *)
        usage
        exit 1
        ;;
esac
