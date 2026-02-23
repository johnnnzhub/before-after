#!/bin/bash
cd "$(dirname "$0")"
STREAMLIT="$HOME/Library/Python/3.9/bin/streamlit"
if ! command -v streamlit &> /dev/null && [ -f "$STREAMLIT" ]; then
  exec "$STREAMLIT" run app.py --server.address=0.0.0.0 --server.headless=true
else
  exec streamlit run app.py --server.address=0.0.0.0 --server.headless=true
fi
