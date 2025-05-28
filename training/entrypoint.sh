#!/bin/bash

export PREFECT_API_URL=http://0.0.0.0:4200/api
prefect config set PREFECT_API_URL=$PREFECT_API_URL

# Start Prefect server
prefect server start --host 0.0.0.0 &
sleep 25

# Create the work pool if it doesn't exist
prefect work-pool create default --type process || true

# Deploy the flow directly
prefect deploy
sleep 15

# Start worker (foreground process)
prefect worker start --pool 'default'
