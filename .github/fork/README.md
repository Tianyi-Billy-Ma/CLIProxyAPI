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
  that created them, kept 24 hours after finishing, and at most 32 are unfinished at once
  (429 beyond). Only 4 are sent upstream at a time; the rest stay `queued`.
- Each unfinished job keeps its whole upload in memory, so give the container room
  (e.g., 2G rather than 512M when clients upload many 15-minute chunks at once).

Config example (spark serves MOSS-Transcribe-Diarize as `SPARK-ASR`):

```yaml
api-keys:
  openai-compatibility:
    - name: "spark"
      base-url: "http://100.109.19.25:8001/v1"
      keys:
        - api-key: "<any placeholder>"
      models:
        - name: "SPARK-ASR"

# MOSS replaces its built-in "transcribe + speaker" instruction with `prompt`,
# so a bare hot-word list breaks diarized_json (HTTP 400). This rule wraps any
# non-empty client prompt in the default instruction; no code change needed.
payload:
  replace:
    - models:
        - name: "SPARK-ASR"
          protocol: "openai"
      target:
        paths: ["prompt"]
      replacements:
        - pattern: '(?s)^\s*(\S.*)$'
          replacement: '请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。热词提示：${1}'
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
