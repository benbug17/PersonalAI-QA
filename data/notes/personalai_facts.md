# PersonalAI project facts

Groq retired the model llama-3.1-8b-instant on 16 August 2026. Every model call then failed with a 404 model_not_found error. The model name now comes from the GROQ_MODEL setting and defaults to openai/gpt-oss-20b.

Uploads larger than 10 MB are rejected with a 413 status code. Empty uploads get a 400. Questions longer than 2000 characters get a 422.

Passwords longer than 72 bytes are rejected because bcrypt cannot use more than 72 bytes. Passwords are hashed with bcrypt and never stored as plain text.

The evaluation baseline floors use a tolerance of 15 percent. It was 10 percent before, which let one unlucky answer in the smallest group of 8 cases fail the build.

The audio endpoint only serves file names that are exactly 64 hex characters. This blocks path traversal.

Login tokens are kept in memory, so restarting the server logs everyone out. There is no rate limit on login attempts yet.
