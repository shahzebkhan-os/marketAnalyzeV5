#!/bin/bash

# Configuration
SCRIPT_PATH="$(pwd)/ops/backup_db.sh"
CRON_SCHEDULE="0 0 * * *" # Daily at midnight

# Verify script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Backup script not found at $SCRIPT_PATH"
    exit 1
fi

# Ensure it is executable
chmod +x "$SCRIPT_PATH"

# Add to crontab if not already there
(crontab -l 2>/dev/null; echo "$CRON_SCHEDULE $SCRIPT_PATH >> $(pwd)/logs/backup.log 2>&1") | crontab -

echo "Cron job added: Daily backup at midnight."
echo "Command: $CRON_SCHEDULE $SCRIPT_PATH"
