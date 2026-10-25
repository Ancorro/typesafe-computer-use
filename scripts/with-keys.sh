#!/bin/bash
# Run one command with this repo's Typesafe key from the fleet's encrypted infra secrets (SECRET_NAME below).
# .env keeps the non-secret settings; its TYPESAFE_API_KEY= stays blank since 2026-10-25.
#   scripts/with-keys.sh uv run clicker "open the settings page"
SECRET_NAME=TYPESAFE_STACK_API_KEY
infra=${INFRA:-$HOME/infra}
[ -x "$infra/bin/secrets.sh" ] || infra=/mnt/c/Users/Steven/Desktop/projects/infra
exec "$infra/bin/secrets.sh" run --map "TYPESAFE_API_KEY=$SECRET_NAME" "$@"
