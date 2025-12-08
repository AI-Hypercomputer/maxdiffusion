#!/bin/bash
# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}kubectl get pods | grep $RUN_NAME${NC}"
kubectl get pods | grep $RUN_NAME