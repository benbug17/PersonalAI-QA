# Fixing the Einstein hallucination without fooling yourself

## What we know
- The assistant invents a 1962 Einstein speech (he died in 1955). It fails on every repeat.
- Your SYSTEM_PROMPT already says: "If you do not recognise a person, book, function, or event, or the question has
  a false premise, say so plainly." So the missing piece is not that rule. The model *does* recognise Einstein, so the
  "do not recognise" part never triggers, and it never checks whether the date makes the event possible.
- The candidate rule (`apply_premise_check.py`) asks for exactly that check, in general terms (any named person,
  date, technology), without naming Einstein.

This is a hypothesis. One failing case is thin evidence, so the protocol below decides it by measurement.

## The two new sets (8 cases each, never used to tune anything)
- `evals/halluc_tune_cases.json`: you may iterate on the prompt against this one.
- `evals/halluc_confirm_cases.json`: baseline once, then ONE final run. Never use it to decide what to change.
Both contain dated claims about people (the failure class) plus controls: invented book/paper, functions that do not
exist, a false-premise fact, an event that has not happened. Controls should NOT change; if they get worse, the rule is
doing harm. Every case has two correct answers and two fabrications worded differently, checked by the unit tests.

## Success criteria (written BEFORE any run; do not edit afterwards)
1. Confirm set: the number of cases that pass on all 3 repeats rises by at least 2 versus its baseline.
2. Dev suite stays 42/42.
3. Held-out v1: no case that passed before now fails (the Einstein case may flip to pass).
4. The control cases in the confirm set do not get worse.
If the baseline already passes all 8 confirm cases there is no headroom; stop, the change is not needed for that class.

## Steps (estimated tokens assume about 0.5k per call; check the token line each run)
**0. Install and gate** the files from this package, then `python3.11 -m pytest --cov --cov-fail-under=95`.

**1. Freeze both sets and measure the CURRENT prompt (about 25k tokens):**
```
python3.11 -m evals.heldout_run --cases evals/halluc_tune_cases.json --freeze --repeat 3 --out evals/recorded/halluc_tune_baseline.json
python3.11 -m evals.heldout_run --cases evals/halluc_confirm_cases.json --freeze --repeat 3 --out evals/recorded/halluc_confirm_baseline.json
```
Commit the two `*_manifest.json` files and the two recorded results NOW, before touching the prompt, and paste the
success criteria above into the commit message. Never run `--freeze` on these files again.

**2. Apply the candidate and measure the tune set (about 12k tokens per run):**
```
python3 apply_premise_check.py
python3.11 -m evals.heldout_run --cases evals/halluc_tune_cases.json --repeat 3 --out evals/recorded/halluc_tune_after.json
```
If the tune set does not improve you may reword the rule, at most twice, using only tune-set failures. Re-run the tune
set after each rewording. Do not look at confirm results to decide anything.

**3. Regression checks (about 35k tokens at repeat 1; run on a day with quota left):**
```
python3.11 -m evals.run --repeat 1
python3.11 -m evals.heldout_run --repeat 3 --out evals/recorded/heldout_v1_after_premise_rule.json
```
Do NOT add `--freeze` to the second command. If the dev suite shows any FLAKY or FAIL, run it with `--repeat 3`.

**4. The single confirmation run (about 12k tokens):**
```
python3.11 -m evals.heldout_run --cases evals/halluc_confirm_cases.json --repeat 3 --out evals/recorded/halluc_confirm_final.json
```

**5. Decide by the criteria above, and report either outcome.**
- All four met: commit the `groq_client.py` change and the recorded files, and update the README "Known weaknesses".
- Not met: `python3 apply_premise_check.py --undo`, and record "tried, did not help" in the README. That is still a
  result worth showing: it demonstrates a hypothesis tested and rejected.
After step 4 the confirm set is spent for this prompt version. Any further prompt change needs a new confirm set.

## Limits to state honestly
- Eight cases per set: one case is 12.5 points, and "all 3 repeats" is strict. Treat small differences as noise.
- The tune set and the failure class overlap on purpose; only the confirm set measures generalisation.
- I could not run any of this (no API key). The candidate rule has not been measured.
- Quota: the whole plan is roughly 105k to 150k tokens; spread it over two days if your daily limit is 200k.
