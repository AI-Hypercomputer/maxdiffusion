#!/bin/bash

# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}xpk workload list --cluster=bodaborg-tpu7x-128 --project=cloud-tpu-multipod-dev --zone=us-central1 --filter-by-job=$RUN_NAME${NC}"
xpk workload list --cluster=bodaborg-tpu7x-128 --project=cloud-tpu-multipod-dev --zone=us-central1 --filter-by-job=$RUN_NAME