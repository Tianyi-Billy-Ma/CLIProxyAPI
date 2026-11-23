# Fork notes (Tianyi-Billy-Ma/CLIProxyAPI)

This fork tracks upstream `router-for-me/CLIProxyAPI` and adds one feature:
`POST /v1/audio/transcriptions` (OpenAI multipart) forwarded to an
`openai-compatibility` upstream, with optional async jobs.

## Audio transcriptions

- Sync: the multipart form is passed through unchanged except `model`, which is
  mapped from the alias to the upstream name. The upstream response is returned as is.
- Async: send `Prefer: respond-async`. The answer is `202 {"id","status":"queued"}`.
  Poll `GET /v1/audio/transcriptions/{id}` for `queued`, `processing`,
  `completed` (result fields merged in) or `failed` (`error.message`, `error.code`).
  Unknown, foreign or expired ids answer 404.
- Jobs live in memory only: they are lost on restart, visible only to the API key
  that created them, kept 24 hours after finishing, and at most 32 run at once (429 beyond).

Config example:

```yaml
api-keys:
  openai-compatibility:
    - name: "spark"
      base-url: "http://<spark tailscale ip>:8001/v1"
      keys:
        - api-key: "<upstream key or any placeholder>"
      models:
        - name: "OpenMOSS-Team/MOSS-Transcribe-Diarize"
          alias: "SPARK-ASR"
```

## Workflows

- `fork-test`: build plus `.github/fork/test.sh` on pull requests; the full upstream
  suite runs informationally because upstream has failing tests of its own.
- `fork-image`: on push to `main` (or manual run) builds `linux/arm64` and pushes
  `ghcr.io/tianyi-billy-ma/cli-proxy-api:<upstream>-asr.<run>`, `sha-<commit>` and,
  from `main` only, `latest`. `<upstream>` comes from `.github/fork/upstream-version`.
- `fork-upstream-sync`: Mondays 09:17 UTC (or manual) merges upstream `main` with
  `--no-ff`, tests the result, pushes without force, then starts `fork-image`.
  Conflicts or failures open an `upstream-sync` issue instead.

Upstream's own workflows (`release`, `docker-image`, `auto-retarget-main-pr-to-dev`,
`agents-md-guard`) should be disabled in the Actions tab of this fork.
