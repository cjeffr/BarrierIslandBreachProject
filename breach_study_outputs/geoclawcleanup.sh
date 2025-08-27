#!/usr/bin/env bash
# cleanup.sh
# Remove unwanted build/data/log files but keep/rename breach.data

set -euo pipefail

# 1. Remove all .data files except breach.data
find . -type f -name "*.data" ! -name "breach.data" -exec rm -v {} +

# 2. Rename breach.data to breaches.input if it exists
if [ -f breach.data ]; then
    mv -v breach.data breaches.input
fi

# 3. Remove all .o files
find . -type f -name "*.o" -exec rm -v {} +

# 4. Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rv {} +

# 5. Remove all .log files
find . -type f -name "*.log" -exec rm -v {} +

# 6. Remove xgeoclaw binary if present
find . -type f -name "xgeoclaw" -exec rm -v {} +

