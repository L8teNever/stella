# Stella

Independent **voice / phone** stack. Ida dispatches outbound call **jobs** over MCP. Stella also **answers inbound PSTN** to `TELNYX_FROM_NUMBER` (no MCP). Talks with **Gemini Live** (`GEMINI_API_KEY`) over **Telnyx Call Control**. Stella does **not** share Ida’s live chat, memory, or other MCPs — only `brief` + optional `context` from the job. If the other party asks something that is not in that payload, Stella says she does not know. The one exception is the optional [`frag_ida`](#frag-ida-claude-code-im-anruf) lookup, available only when the **owner** is on the line (outbound dest or inbound caller).

```
Ida / Cursor  --MCP-->  Stella HTTP
                            |  POST /v2/calls (outbound dial + stream_url)
                            |  inbound: call.initiated → answer (stream_url once)
                            |  webhooks /webhooks/telnyx  → SQLite status
                            |  WS /media/{job_id}  <-->  Gemini Live
```

## Run (Docker)

```bash
cp .env.example .env
# fill Telnyx + GEMINI_API_KEY
docker compose up --build
```

Health: `GET http://localhost:8080/health`

Telnyx must be able to reach Stella. Locally, put a tunnel in front (e.g. cloudflared) and set:

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
| `GEMINI_API_KEY` | Gemini Live (voice provider) |

Optional: `STELLA_CALLBACK_URL` (POST JSON when a call hangs up), `STELLA_MCP_TOKEN` (Bearer for `/mcp`).

`STELLA_SKIP_WEBHOOK_VERIFY=true` is **only** for local tests without Telnyx signatures. Do not use it in any shared environment.

No secrets belong in git.

## Gemini Live

`GET /health` reports `voice_provider.primary` (`gemini`). If `GEMINI_API_KEY` is empty, `stella_call` fails with an explicit error before dialing.

Telnyx streams **PCMU 8 kHz** (20 ms / 160-byte RTP frames). Gemini Live wants **PCM 16-bit LE / 16 kHz in** and typically **24 kHz PCM out**, so Stella box-filter downsamples, μ-law-encodes, and emits aligned 20 ms frames on the bridge. After a job is created, `stella_call_status` includes `voice_provider` (always `gemini`).

Optional overrides: **`GEMINI_LIVE_MODEL`** (default **`gemini-3.8-live`**, confirmed Live / native-audio BidiGenerateContent id — [model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live)). Do **not** set `gemini-3.8-live-extended-thinking` (extra reasoning latency). The host may pin the same default with `GEMINI_LIVE_MODEL=gemini-3.8-live`. `GEMINI_VOICE` defaults to `Aoede` (female). `GEMINI_THINKING_BUDGET` defaults to `0` (thinking off): 2.5 and 3.8 Live get `thinkingBudget: 0`; 3.1 Live gets `thinkingLevel: minimal` only (the API rejects setup if both are set, and 3.8 rejects `thinkingLevel`). Set `-1` to omit `thinkingConfig`.

**Turn latency (Gemini):** By default Stella uses **client RMS VAD** (`STELLA_CLIENT_VAD=true`): inbound PCMU is energy-gated locally; after ~`STELLA_CLIENT_VAD_SILENCE_MS` (default 120) of quiet following speech, Stella sends `realtimeInput.activityEnd` and Gemini starts the reply without waiting for Google’s automatic end-of-speech (often ~1–2s+). Automatic VAD is disabled in that mode. Set `STELLA_CLIENT_VAD=false` to use Google’s detector (`GEMINI_VAD_SILENCE_DURATION_MS`, `GEMINI_VAD_START_SENSITIVITY` / `END_SENSITIVITY`, `GEMINI_VAD_PREFIX_PADDING_MS`). Barge-in still uses `START_OF_ACTIVITY_INTERRUPTS`; Telnyx `clear` is sent only on Gemini `interrupted`, not on every energy blip.

`STELLA_LATENCY_LOG=true` logs `setup_to_first_audio_ms` and `silence_to_first_audio_ms` (activityEnd → first outbound PCMU). Remaining delay after activityEnd is Gemini first-audio TTFT (typically a few hundred ms to ~1s+); that floor is not fully removable in-app.

The in-repo default is **`gemini-3.8-live`**, not 3.1 Flash Live and not any 2.5 native-audio preview. If that model’s Live **setup** handshake fails, Stella retries (same WebSocket path only): `gemini-3.1-flash-live-preview` → `gemini-2.5-flash-native-audio-preview-12-2025` → `gemini-2.5-flash-native-audio-latest`. It never uses `gemini-2.5-flash-native-audio-preview-09-2025` or `gemini-3.8-live-extended-thinking`. Do not use translate/transcribe Live ids for PSTN dialogue.

**Hang-up:** The model must call `hang_up` to drop the PSTN leg (spoken “tschüss” is not enough). TelnyxMediaGuard still waits for stream start and outbound playout. A backup fires if the spoken transcript looks like a farewell and inbound stays quiet for `STELLA_FAREWELL_HANGUP_S` (disable with `STELLA_FAREWELL_HANGUP=false`).

## Frag Ida (Claude Code im Anruf)

On a call to the **owner number**, Gemini can look up Simon’s personal data (calendar, timetable, homework, mail read-only, smart-home status) via the function tool `frag_ida`:

1. Simon asks; Gemini says “Moment, ich schau nach” and calls `frag_ida(frage, bestaetigt?)`.
2. Stella runs Claude Code headless (`claude -p`, model `ASK_IDA_MODEL`, default `sonnet`) in the container with only the MCP servers from `ASK_IDA_MCP_CONFIG` (`--strict-mcp-config`). Built-in tools (shell, files, web) are off; the question goes in via stdin, never through a shell.
3. Claude answers in 1–3 spoken German sentences; the text (max 600 chars) goes back to Gemini as the tool result and Gemini says it.

The call is handled in a background task, so audio, VAD, hang-up and farewell guards keep running. `gemini-3.8-live` is not documented to support non-blocking function calls (only 2.5 Flash Live is), so Stella uses the default blocking behaviour and relies on Gemini announcing the lookup before the call.

**Security model**

- `frag_ida` is offered only if `ASK_IDA_ENABLED=true` **and** the party on the line equals `STELLA_OWNER_NUMBER` (outbound dest or inbound caller; stored as `allow_ida`, re-checked when the media session starts). `stella_call`/`stella_briefing_call` accept `allow_ida=false` to switch it off; `true` never overrides the number check. Calls with anyone else get no such tool and Stella says she doesn’t know.
- Only tools in `ASK_IDA_ALLOWED_TOOLS` (exact `mcp__<server>__<tool>` names, **read-only**) are usable; anything else is denied by Claude Code (`--permission-mode dontAsk`). `ASK_IDA_DENY_TOOLS` always wins (default: SSH, Cloudflare, delete/clear tools).
- Optional `ASK_IDA_WRITE_TOOLS` (empty by default) are only unlocked when Gemini passes `bestaetigt=true`, which it must do only after Simon explicitly says yes to a spoken “Soll ich das wirklich machen?”. This confirmation is model-enforced, not cryptographic; keep the list empty unless you accept that.
- Prompt injection (e.g. a malicious mail) is mitigated by the read-only allowlist and the system prompt, not eliminated.

**Setup**

1. `cp ida-mcp.example.json data/ida-mcp.json` and fill in your Ida MCP servers (URLs, `${IDA_MCP_TOKEN}`). Never commit it.
2. In `.env`: `ASK_IDA_ENABLED=true`, `STELLA_OWNER_NUMBER=+49…`, `ASK_IDA_ALLOWED_TOOLS=mcp__Ida_Untis__stundenplan,…`, and Claude Code auth: `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`), plus `IDA_MCP_TOKEN`.
3. `docker compose up -d --build` (the image installs Node + Claude Code, version pinned in the `Dockerfile`; `HOME`/`CLAUDE_CONFIG_DIR` live under `/data`).

**Latency:** a lookup takes several seconds (Claude start + MCP + model; ~4–6 s measured locally with one tool). With `STELLA_LATENCY_LOG=true`, Stella logs `stella_latency ask_ida_ms ms=<n>`. Timeout: `ASK_IDA_TIMEOUT_S` (default 25 s).

## Telnyx

1. Create a Call Control application / Voice API connection.
2. Assign a number (`TELNYX_FROM_NUMBER`).
3. Copy API key, connection id, and webhook public key.
4. Point the connection’s webhook at `{STELLA_PUBLIC_BASE_URL}/webhooks/telnyx` (Stella also sends `webhook_url` on each dial).

Stella dials E.164, starts **bidirectional media streaming** (PCMU or PCMA at 8 kHz; PCMA for German +49 PSTN) into `/media/{job_id}`, and bridges that socket to Gemini Live. Call events (`call.initiated`, `call.answered`, `call.hangup`, streaming failures) are persisted and returned from MCP `stella_call_status`.

### Inbound (dial Stella)

When someone calls `TELNYX_FROM_NUMBER` (e.g. `+4973613809988`), Telnyx sends `call.initiated` with `direction: incoming`. Stella creates a `kind=inbound` job (`to` = caller) and **answers with `stream_url` once** — same bidirectional fields as outbound dial (`stream_track=inbound_track` = remote-party audio, PCMA on DE). Do **not** also call `streaming_start`; that second connect returns Telnyx 422/90046 and can leave Gemini deaf to the caller.

| Variable | Default | Meaning |
| --- | --- | --- |
| `STELLA_INBOUND_ENABLED` | `true` | Master switch. `false` hangs up every inbound call. |
| `STELLA_OWNER_NUMBER` | empty | Simon’s E.164. Always answered when inbound is on. |
| `STELLA_INBOUND_UNKNOWN` | `hangup` | Other callers: `hangup`, `speak` (play `STELLA_INBOUND_REJECT_TEXT` then hang up), or `answer` (talk, **no** `frag_ida`). |
| `STELLA_INBOUND_BRIEF` | German chat with Simon | Gemini brief for owner inbound. |

Inbound is **automatic** — no MCP tool to “pick up”. Use `stella_recent_calls` / `stella_call_status` to see jobs. Point the Call Control connection webhook at `/webhooks/telnyx` (same as outbound).

## MCP (Ida / Cursor)

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

**`stella_call_status`** — `call_id` from the place-call response or an inbound job. Returns status, outcome, transcript, Telnyx events.

**`stella_recent_calls`** — last jobs (`limit`, optional `kind`: `call` / `briefing` / `inbound`). Inbound appears here when someone dials Stella.

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
export STELLA_DB_PATH=./data/stella.db
stella serve
```

Tests: `pytest`.

## Out of scope

Production deploy, shared live context with Ida, mid-call Ida lookups.
