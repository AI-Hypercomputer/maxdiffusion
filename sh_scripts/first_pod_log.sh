#!/bin/bash
# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}kubectl logs $(kubectl get pods | grep $RUN_NAME | head -1 | awk '{print $1}') --all-containers --tail=200${NC}"
kubectl logs $(kubectl get pods | grep $RUN_NAME | head -1 | awk '{print $1}') --all-containers --tail=200