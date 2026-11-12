# Engineering items 10 to 14: what changed and how to use it

Verified here: 850 tests pass, 100% coverage, ruff clean. Not verified: anything that needs your API key.
Copy each file to the same path in your repo (the zip is laid out that way), then run
`python3.11 -m pytest --cov --cov-fail-under=95`.

## 11. The agent eval now reports repairs and restarts
`python3.11 -m evals.agent_run --embedder onnx --repeat 3` prints, per case, ", N repaired" / ", N restarted" when
they happened, and a total line: "Malformed tool calls repaired in place: X; whole-question restarts: Y".
Both counts are saved per attempt in the results file.

## 13. /agent rate limit, and /metrics is now off by default  (BEHAVIOUR CHANGE)
- `/agent`: 10 requests per user per 60 s by default. Over the limit returns 429 with a `Retry-After` header and costs
  no model tokens. Settings, optional, in `.env`:
  ```
  # AGENT_RATE_LIMIT=10               # requests per user per window; 0 turns the limit off
  # AGENT_RATE_WINDOW_SECONDS=60
  # METRICS_TOKEN=                    # set a long random value to enable GET /metrics
  ```
- `/metrics` previously worked for any logged-in user. It now returns 404 unless `METRICS_TOKEN` is set, and then only
  for callers sending the header `X-Metrics-Token: <token>`. Create a token with
  `python3.11 -c "import secrets; print(secrets.token_urlsafe(32))"`, then
  `curl -H "X-Metrics-Token: $METRICS_TOKEN" http://localhost:8000/metrics`.
- The limiter is in memory and per process: several server workers each keep their own count, and a restart resets it.

## 12. Why is an agent model call about 9x a plain call? (the probe; needs your key)
```
python3.11 -m evals.latency_probe --rounds 6        # about 28 calls; I estimate 20k to 25k tokens
```
Four arms on the same question, interleaved so provider load hits them equally: `plain` (your client), `lc_plain`
(LangChain client, same prompt), `lc_prompt` (the agent's longer prompt), `lc_tools` (prompt plus tool schemas).
Read the "p50 ratios" and "ms per completion token". If one ratio is far above 1, that is the step to investigate;
if completion tokens differ a lot between arms, the time is generation, not overhead. Send me the output.

## 14. Comparing two judge models (offline)
```
JUDGE_MODEL=<other model> python3.11 -m evals.judge_audit audit evals/recorded/heldout_run3_judge.json --out evals/recorded/judge_audit_modelB.json
python3.11 -m evals.judge_audit compare evals/recorded/judge_audit_run1.json evals/recorded/judge_audit_modelB.json
```
`compare` needs no key. It reports how often the two judges agree with each other and with the keyword checks, and lists
every disagreement. `audit` also takes `--cases PATH` for results from another cases file.

## 10. The Einstein hallucination
See HALLUCINATION_PROTOCOL.md. New: `evals/halluc_tune_cases.json`, `evals/halluc_confirm_cases.json`, and
`heldout_run --cases PATH` (each file gets its own freeze manifest). The candidate prompt rule is in
`apply_premise_check.py`, which is NOT part of the repo and changes nothing until you run it.

## Files
Changed: agent.py, agent_router.py, api.py, evals/agent_run.py, evals/heldout_run.py, evals/judge_audit.py,
tests/api/test_agent_router.py, tests/evals_cli/test_{agent_cli,heldout_cli,judge_audit_cli}.py, tests/unit/test_agent.py.
New: evals/latency_probe.py, evals/halluc_{tune,confirm}_cases.json, tests/api/test_agent_router_limits.py,
tests/evals_cli/test_latency_probe_cli.py, tests/genai/test_hallucination_sets.py, tests/unit/test_rate_limiter.py.
