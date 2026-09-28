# Stella

Independent **voice / phone** stack. Ida (or Grok Bot) dispatches a call **job** over MCP. Stella places the call with **Telnyx Call Control** and talks with **Grok Voice** (primary). If Grok realtime fails, Stella falls back to **Gemini Live** using `GEMINI_API_KEY`. Stella does **not** share Ida’s live chat, memory, or other MCPs — only `brief` + optional `context` from the job. If the other party asks something that is not in that payload, Stella says she does not know. There is no mid-call roundtrip to Ida.

```
Ida / Cursor  --MCP-->  Stella HTTP
                            |  POST /v2/calls (Telnyx dial + stream_url)
                            |  webhooks /webhooks/telnyx  → SQLite status
                            |  WS /media/{job_id}  <-->  Grok realtime (primary)
                            |                       or Gemini Live (fallback)
```

## Run (Docker)

```bash
cp .env.example .env
# fill Telnyx + XAI_API_KEY (or complete OAuth after the container is up)
# optional: GEMINI_API_KEY so a Grok realtime 403 can fall back to Gemini Live
docker compose up --build
```

Health: `GET http://localhost:8080/health`

Telnyx must be able to reach Stella. Locally, put a tunnel in front (ngrok / cloudflared) and set:

```
STELLA_PUBLIC_BASE_URL=https://your-tunnel.example
```

Webhook path: `{STELLA_PUBLIC_BASE_URL}/webhooks/telnyx`  
Media stream: `wss://…/media/{job_id}` (derived automatically).

## Environment

See `.env.example`. Required for a real outbound call:

| Variable | Purpose |
| --- | --- |
| `TELNYX_API_KEY` | Call Control API |
| `TELNYX_CONNECTION_ID` | Voice API connection / Call Control App |
| `TELNYX_FROM_NUMBER` | E.164 caller ID |
| `TELNYX_PUBLIC_KEY` | Ed25519 public key to verify webhooks |
| `STELLA_PUBLIC_BASE_URL` | Public HTTPS origin Telnyx can hit |
| `XAI_API_KEY` **or** SuperGrok OAuth tokens | Grok Voice (primary) |
| `GEMINI_API_KEY` | Gemini Live fallback when Grok session connect fails |

Optional: `STELLA_CALLBACK_URL` (POST JSON when a call hangs up), `STELLA_MCP_TOKEN` (Bearer for `/mcp`).

`STELLA_SKIP_WEBHOOK_VERIFY=true` is **only** for local tests without Telnyx signatures. Do not use it in any shared environment.

No secrets belong in git.

## xAI / SuperGrok OAuth

Same device-code pattern as Warp / Hermes (`auth.x.ai`, public Grok CLI `client_id`). Tokens are stored in the `stella-data` volume (`XAI_OAUTH_TOKEN_PATH`).

Inside the container:

```bash
docker compose exec stella stella oauth login
# open the printed URL, approve, wait until tokens are saved
docker compose exec stella stella oauth status
```

Or HTTP:

1. `POST /oauth/xai/start`
2. Approve in the browser
3. `POST /oauth/xai/poll` with `{ "device_code": "..." }`

If both OAuth tokens and `XAI_API_KEY` exist, **OAuth wins** (subscription quota). `XAI_API_KEY` is the documented local/dev fallback.

Voice model default: `grok-voice-latest` (`XAI_VOICE_MODEL`). Voice: `XAI_VOICE=eve`.

## Gemini Live fallback (sticky per call)

Grok remains the primary realtime path. When MCP starts a call (`stella_call` / `stella_briefing_call`), Stella probes Grok Voice realtime **once**. Success → **Grok for the entire call**. Failure (HTTP 403, auth, quota/limit, connect) → **Gemini Live for the entire call**. That choice is stored on the job (`voice_provider`) and reused if Telnyx reconnects `/media/{job_id}`. There is no mid-call re-probe and no fallback after the session is already running.

Grok-down cache (bonus, in-memory, this process only): a failed Grok probe is remembered for **`GROK_DOWN_CACHE_TTL_SECONDS` = 300 (5 minutes)** so the next MCP call does not pay the 403 handshake again. After the TTL, Stella probes Grok again. `GET /health` reports `voice_provider.sticky`, `grok_down_cache_ttl_seconds`, and `grok_down_cached`.

Telnyx streams **PCMU 8 kHz** (20 ms / 160-byte RTP frames). Gemini Live wants **PCM 16-bit LE / 16 kHz in** and typically **24 kHz PCM out**, so Stella box-filter downsamples, μ-law-encodes, and emits aligned 20 ms frames on the bridge. After a job is created, `stella_call_status` includes `voice_provider` (`grok` or `gemini`).

If Grok cannot be used and `GEMINI_API_KEY` is empty, the job fails with an explicit error before dial (or the media session fails if only Gemini was locked).

Optional overrides: **`GEMINI_LIVE_MODEL`** (default **`gemini-3.8-live`**, confirmed Live / native-audio BidiGenerateContent id — [model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live)). Do **not** set `gemini-3.8-live-extended-thinking` (extra reasoning latency). The host may pin the same default with `GEMINI_LIVE_MODEL=gemini-3.8-live`. `GEMINI_VOICE` defaults to `Aoede` (female). Grok remains `XAI_VOICE=eve`. `GEMINI_THINKING_BUDGET` defaults to `0` (thinking off): 2.5 Live gets `thinkingBudget: 0`; 3.x Live also sends `thinkingLevel: minimal`. Set `-1` to omit `thinkingConfig`.

**Turn latency (Gemini):** By default Stella uses **client RMS VAD** (`STELLA_CLIENT_VAD=true`): inbound PCMU is energy-gated locally; after ~`STELLA_CLIENT_VAD_SILENCE_MS` (default 120) of quiet following speech, Stella sends `realtimeInput.activityEnd` and Gemini starts the reply without waiting for Google’s automatic end-of-speech (often ~1–2s+). Automatic VAD is disabled in that mode. Set `STELLA_CLIENT_VAD=false` to use Google’s detector (`GEMINI_VAD_SILENCE_DURATION_MS`, `GEMINI_VAD_START_SENSITIVITY` / `END_SENSITIVITY`, `GEMINI_VAD_PREFIX_PADDING_MS`). Barge-in still uses `START_OF_ACTIVITY_INTERRUPTS`; Telnyx `clear` is sent only on Gemini `interrupted`, not on every energy blip.

`STELLA_LATENCY_LOG=true` logs `setup_to_first_audio_ms` and `silence_to_first_audio_ms` (activityEnd → first outbound PCMU). Remaining delay after activityEnd is Gemini first-audio TTFT (typically a few hundred ms to ~1s+); that floor is not fully removable in-app.

The in-repo default is **`gemini-3.8-live`**, not 3.1 Flash Live and not any 2.5 native-audio preview. If that model’s Live **setup** handshake fails, Stella retries (same WebSocket path only): `gemini-3.1-flash-live-preview` → `gemini-2.5-flash-native-audio-preview-12-2025` → `gemini-2.5-flash-native-audio-latest`. It never uses `gemini-2.5-flash-native-audio-preview-09-2025` or `gemini-3.8-live-extended-thinking`. Do not use translate/transcribe Live ids for PSTN dialogue.

**Hang-up:** The model must call `hang_up` to drop the PSTN leg (spoken “tschüss” is not enough). TelnyxMediaGuard still waits for stream start and outbound playout. A backup fires if the spoken transcript looks like a farewell and inbound stays quiet for `STELLA_FAREWELL_HANGUP_S` (disable with `STELLA_FAREWELL_HANGUP=false`).

## Telnyx

1. Create a Call Control application / Voice API connection.
2. Assign a number (`TELNYX_FROM_NUMBER`).
3. Copy API key, connection id, and webhook public key.
4. Point the connection’s webhook at `{STELLA_PUBLIC_BASE_URL}/webhooks/telnyx` (Stella also sends `webhook_url` on each dial).

Stella dials E.164, starts **bidirectional media streaming** (PCMU 8 kHz) into `/media/{job_id}`, and bridges that socket to the provider chosen at MCP call start (Grok Voice or Gemini Live). Call events (`call.initiated`, `call.answered`, `call.hangup`, streaming failures) are persisted and returned from MCP `stella_call_status`.

## MCP (Ida / Cursor / Grok Bot)

### HTTP (recommended for Docker)

Stella serves Streamable HTTP MCP at:

`http://localhost:8080/mcp`

If `STELLA_MCP_TOKEN` is set, send `Authorization: Bearer <token>`.

Cursor `mcp.json` example:

```json
{
  "mcpServers": {
    "stella": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

### stdio

```json
{
  "mcpServers": {
    "stella": {
      "command": "docker",
      "args": ["compose", "exec", "-T", "stella", "stella-mcp"]
    }
  }
}
```

Or locally after `pip install -e .`: `stella mcp` / `stella-mcp`.

### Tools

**`stella_call`**

| Param | Required | Notes |
| --- | --- | --- |
| `to` | yes | E.164 |
| `brief` | yes | Task / script (this is the agent’s whole world) |
| `context` | no | Extra free text from Ida |
| `speak_to` | no | Human label (“Simon”, “the restaurant”) |

**`stella_call_status`** — `call_id` from the place-call response. Returns status, outcome, transcript, Telnyx events.

**`stella_briefing_call`** — `to` + `text` to read aloud (morning briefing), then hang up.

### Example jobs

- `stella_call(to="+49…", brief="Reserve a table tomorrow 19:00 for 2, name Franz.", speak_to="the restaurant")`
- `stella_briefing_call(to="+49…", text="…morning briefing…", speak_to="Simon")`
- `stella_call(to="+49…", brief="Ask if the appointment still works. Report yes/no/reschedule.", context="Appointment: Tuesday 10:00 with Dr. X.")`

Then poll `stella_call_status`.

## Local (no Docker)

Python 3.12+:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export STELLA_DB_PATH=./data/stella.db XAI_OAUTH_TOKEN_PATH=./data/xai_oauth.json
stella serve
```

Tests: `pytest`.

## Out of scope

Production deploy, shared live context with Ida, mid-call Ida lookups.
