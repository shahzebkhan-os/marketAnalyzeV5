#!/bin/bash

# Configuration
BACKUP_DIR="./backups"
DB_NAME="holographic_glenn"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FILENAME="$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

# Ensure backup directory exists
mkdir -p $BACKUP_DIR

# Perform Backup
echo "Starting backup for $DB_NAME..."
pg_dump $DB_NAME | gzip > $FILENAME

if [ $? -eq 0 ]; then
  echo "Backup successful: $FILENAME"
  
  # Retention Policy: Keep last 7 days
  find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete
else
  echo "Backup failed!"
  exit 1
fi
