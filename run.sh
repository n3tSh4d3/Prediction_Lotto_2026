#!/usr/bin/env bash
# Avvia il server web su http://localhost:8010
cd "$(dirname "$0")"
exec python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8010 "$@"
