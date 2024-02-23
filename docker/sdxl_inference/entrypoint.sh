#!/bin/bash
export PORT=$AIP_HTTP_PORT
uvicorn handler:app --proxy-headers --host 0.0.0.0 --port $PORT