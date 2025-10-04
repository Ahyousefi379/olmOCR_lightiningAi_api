#!/usr/bin/env bash
# run_server.sh - restart loop wrapper
# Make this executable: chmod +x run_server.sh

LOGFILE="./server_run.log"

while true; do
  echo "$(date) - Starting server..." | tee -a "$LOGFILE"
  # Run the server; stdout/stderr go into server_run.log
  python3 server.py 2>&1 | tee -a "$LOGFILE"
  EXIT_CODE=$?
  echo "$(date) - Server exited with code $EXIT_CODE. Restarting in 5s..." | tee -a "$LOGFILE"
  sleep 5
done
