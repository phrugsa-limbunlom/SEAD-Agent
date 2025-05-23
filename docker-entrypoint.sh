#!/bin/bash

if [[ "$1" == "python" && "$2" == "/home/src/tests/run_tests.py" ]]; then
    exec "$@"
else
    # Run FastAPI app on port 8080
    exec uvicorn src.main:app --host 0.0.0.0 --port 8080
fi