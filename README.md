# Voice-First Learning Assistant

A Python-only voice learning assistant built with Streamlit. Uses speech recognition, LLM chat, and text-to-speech for a complete voice interaction experience.

## Features

- **Voice Input**: Record questions using your microphone
- **Speech Recognition**: Automatic transcription using Whisper (with speech_recognition fallback)
- **AI Assistant**: Powered by Groq's openai/gpt-oss-20b model (configurable via `GROQ_MODEL`)
- **Voice Output**: Text-to-speech responses using gTTS
- **User Authentication**: Secure login with bcrypt password hashing
- **Conversation History**: SQLite database stores all interactions
- **TTS Caching**: Hash-based caching to avoid regenerating identical audio

## Tech Stack

- **UI**: Streamlit + audio-recorder-streamlit
- **ASR**: OpenAI Whisper (primary), speech_recognition (fallback)
- **LLM**: Groq API with openai/gpt-oss-20b (set `GROQ_MODEL` to change)
- **TTS**: gTTS (Google Text-to-Speech)
- **Database**: SQLite with bcrypt for password security
- **Language**: 100% Python

## Prerequisites

- Python 3.8 or higher
- ffmpeg (required for audio processing)

### Installing ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd <project-directory>
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_actual_api_key_here
```

Get your Groq API key from: https://console.groq.com/keys

## Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

1. **Create an Account**: Register with a username and password on the first run
2. **Login**: Use your credentials to access the assistant
3. **Record a Question**: Click the microphone button and speak your question
4. **Listen to Response**: The assistant will transcribe, process, and respond with both text and audio
5. **View History**: Check previous conversations in the history section

## FastAPI Backend

`api.py` exposes the STT -> LLM -> TTS pipeline over HTTP (Streamlit UI in `app.py` is unchanged).

```bash
pip install -r requirements-api.txt
uvicorn api:app --reload        # docs at http://localhost:8000/docs
```

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register`, `/auth/login` | create account / get bearer token |
| POST | `/chat` | text question -> LLM answer (`"speak": true` adds TTS audio) |
| POST | `/voice` | audio upload -> transcript + answer + audio URL |
| GET | `/history`, `/audio/{hash}` | saved conversations / cached TTS mp3 |

## Testing

```bash
pip install -r requirements-dev.txt
pytest                                    # full offline suite, no API key needed
pytest --cov                              # with coverage (CI requires >= 95%)
ruff check .                              # lint
RUN_LIVE_LLM=1 GROQ_API_KEY=... pytest -m live   # real-model GenAI evals through pytest
python -m evals.run                       # same evals + JSON report + baseline gate
python -m evals.run --write-baseline      # save measured pass rates + snapshot real answers
```

| Suite | Location | What it proves |
|---|---|---|
| Unit | `tests/unit` | db (hashing, concurrency, injection), tts (cache, markdown-to-speech sanitiser, failures), asr (real Whisper / SpeechRecognition code paths with fake engines, fallback order, temp-file cleanup), Groq request shape and model config |
| API | `tests/api` | every endpoint, response schemas, OpenAPI contract + bearer scheme, auth, pipeline order |
| UI | `tests/ui` | the real Streamlit `app.py` driven through `streamlit.testing` (login, register, voice turn, every error branch, history, logout, cache) |
| Integration | `tests/integration` | STT -> LLM -> DB -> TTS across real modules; persistence across restarts; multi-user isolation |
| Negative | `tests/negative` | bad auth/input, SQL injection, path traversal, oversize upload, STT / LLM / TTS / DB outages |
| GenAI eval | `evals/`, `tests/genai`, `tests/evals_cli` | 40 structured cases, the harness itself, the CLI, and replay of real recorded answers |

### GenAI evaluation design

`evals/cases.json` holds 40 cases in five categories: instruction following (length limit, no-markdown
for voice, analogies, "ignore the limit" requests), relevance, hallucination (nonexistent prize / novel /
function, false premise, impossible future, anachronism), robustness and safety (typos, non-English,
emoji, SQL-like text, prompt injection x2, malware, hate, self-harm), and regression (stable facts).

* **Deterministic validators**, no LLM judge: length, keyword sets, regex, markdown detection, and a
  system-prompt-leak detector that fails on any 6-word run copied from the prompt.
* **No rubber-stamp cases.** Every case ships `pass_examples` and `fail_examples`, and the suite asserts that
  each one passes and fails as declared, that empty/None always fails, and that a generic filler reply
  passes at most 10% of the suite.
* **Regression gate.** Per-category pass rates are compared with `evals/baseline.json`; CI fails if one drops.
* **Replay.** `--write-baseline` snapshots the real model's answers to `evals/recorded/responses.json`;
  `tests/genai/test_replay.py` re-validates them offline on every PR.
* The model is configurable (`GROQ_MODEL`); evals run at temperature 0.

CI (`.github/workflows/ci.yml`) lints and runs the offline suite on every push/PR, and runs the live eval when
the `GROQ_API_KEY` secret is set (also weekly, to catch silent model drift).

### Known limitations

* Sessions are in-memory (restart logs everyone out) and there is no login rate limiting.
* Evals measure observable behaviour with heuristics (keywords/regex), not semantic quality.
* Whisper and gTTS themselves are not exercised in CI, only the code around them.

## Project Structure

```
.
├── app.py                # Streamlit UI (voice recorder, history, login)
├── api.py                # FastAPI backend (STT -> LLM -> TTS over HTTP)
├── db.py                 # SQLite + bcrypt auth, conversation history
├── groq_client.py        # Groq LLM client, system prompt, GROQ_MODEL config
├── asr.py                # Speech recognition (Whisper + fallback)
├── tts.py                # gTTS with hash cache + markdown-to-speech cleaner
├── evals/                # GenAI evaluation: cases, validators, runner, CLI, baseline
├── tests/                # unit, api, ui, integration, negative, genai suites
├── .github/workflows/    # CI: lint, tests + coverage gate, live LLM regression
├── requirements.txt      # Streamlit app deps (Whisper etc.)
├── requirements-api.txt  # FastAPI backend deps
└── requirements-dev.txt  # test + lint deps
```

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required)

### Models

- **ASR**: Whisper "base" model (good balance of speed and accuracy)
- **LLM**: openai/gpt-oss-20b via Groq (override with `GROQ_MODEL`)
- **TTS**: gTTS with English language

## Troubleshooting

### Audio Recording Issues

If the audio recorder doesn't appear or work:
- Ensure your browser has microphone permissions
- Try a different browser (Chrome/Edge work best)
- Check that ffmpeg is installed and in PATH

### Transcription Failures

If Whisper fails to load:
- The app will automatically fall back to speech_recognition
- Ensure you have a stable internet connection (for speech_recognition)
- Check that audio files are in a supported format

### Groq API Errors

If you see "Failed to get response":
- Verify your GROQ_API_KEY is set correctly in .env
- Check your API key is valid at https://console.groq.com/keys
- Ensure you have API credits/quota remaining

### Database Issues

If login/registration fails:
- Delete `app.db` and restart the app to recreate the database
- Check file permissions in the project directory

## Development

### Adding New Features

The modular structure makes it easy to extend:

- **New ASR providers**: Add functions to `asr.py`
- **Different LLMs**: Modify `groq_client.py`
- **Alternative TTS**: Update `tts.py`
- **UI changes**: Edit `app.py`

### Database Schema

**users table:**
- `id`: Primary key
- `username`: Unique username
- `password_hash`: Bcrypt-hashed password
- `created_at`: Timestamp

**history table:**
- `id`: Primary key
- `user_id`: Foreign key to users
- `user_query`: User's question
- `assistant_response`: LLM's response
- `created_at`: Timestamp

## Notes

- First Whisper transcription may take longer as the model downloads
- TTS cache grows over time; use "Clear Cache" button in the sidebar
- All audio processing happens server-side for better reliability
- Session data is stored in Streamlit's session state

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure environment variables are set correctly
4. Check Groq API status and quotas
