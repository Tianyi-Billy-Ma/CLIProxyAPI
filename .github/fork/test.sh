#!/usr/bin/env bash
# Packages touched by the fork's audio transcription patch; these must pass.
set -euo pipefail
go test -race ./sdk/api/handlers/... ./sdk/cliproxy/auth/ ./internal/api/
