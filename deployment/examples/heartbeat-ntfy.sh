#!/bin/bash
# Heartbeat script that publishes hostname to ntfy.sh
#
# Deploy with:
#   python3 run-tasks.py deploy-service \
#       -D SERVICE_NAME=heartbeat \
#       -D LOCAL_FILE_PATH=examples/heartbeat-ntfy.sh
#
# View at: https://ntfy.sh/oYhIC0qan6FDTxsk

TOPIC="oYhIC0qan6FDTxsk"
HOSTNAME=$(hostname)

while true; do
    curl -s -d "Heartbeat from $HOSTNAME at $(date)" "ntfy.sh/$TOPIC"
    sleep 60
done