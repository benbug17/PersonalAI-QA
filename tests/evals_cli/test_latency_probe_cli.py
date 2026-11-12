"""evals.latency_probe: the interleaving, timing and reporting logic, with a fake clock and fake arms."""
import pytest

import groq_client
from evals import latency_probe as probe
from timing import tracker

pytestmark = pytest.mark.genai


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance_ms(self, ms):
        self.now += ms / 1000


class ProviderError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def arm(clock, ms, log, name, completion=10, prompt=100):
    def call(question):
        log.append((name, question))
        clock.advance_ms(ms)
        return {"prompt_tokens": prompt, "completion_tokens": completion}
    return call


N = len(probe.ARMS)                      # the tests follow the real arm list, whatever it contains


def all_arms(clock, log, times=None):
    times = times or tuple(100 * (i + 1) for i in range(N))          # arm i takes 100 * (i + 1) ms
    return {n: arm(clock, ms, log, n) for n, ms in zip(probe.ARMS, times)}


def results_for(ms, prompt=(), completion=(), errors=0, calls=None):
    """A results dict as run_probe builds it."""
    return {"ms": list(ms), "prompt": list(prompt), "completion": list(completion), "errors": errors,
            "calls": calls if calls is not None else [(m, None) for m in ms]}


QS = ["q0", "q1", "q2"]


# ---- run_probe ---------------------------------------------------------------------------------------------------------
def test_each_arm_is_timed_with_the_clock_around_its_own_call():
    clock, log = Clock(), []
    results = probe.run_probe(all_arms(clock, log), QS, rounds=2, warmup=0, clock=clock)
    assert results["plain"]["ms"] == pytest.approx([100.0, 100.0]) and results["lc_tools"]["ms"] == pytest.approx([100.0 * N, 100.0 * N])


def test_the_arm_that_goes_first_rotates_every_round():
    clock, log = Clock(), []
    probe.run_probe(all_arms(clock, log), QS, rounds=3, warmup=0, clock=clock)
    names = [n for n, _ in log]
    arms = list(probe.ARMS)
    for round_number in range(3):
        expected = arms[round_number:] + arms[:round_number]                     # the starting arm moves along by one each round
        assert names[round_number * N:(round_number + 1) * N] == expected


def test_every_arm_gets_the_same_question_within_a_round_and_questions_cycle():
    clock, log = Clock(), []
    probe.run_probe(all_arms(clock, log), QS, rounds=4, warmup=0, clock=clock)
    per_round = [{q for _, q in log[i:i + N]} for i in range(0, 4 * N, N)]
    assert per_round == [{"q0"}, {"q1"}, {"q2"}, {"q0"}]


def test_warmup_rounds_are_run_but_not_recorded():
    clock, log = Clock(), []
    results = probe.run_probe(all_arms(clock, log), QS, rounds=1, warmup=2, clock=clock)
    assert len(log) == N * 3                                              # 2 warm-up rounds + 1 measured, all executed
    assert all(len(r["ms"]) == 1 for r in results.values())               # only the measured one is recorded
    assert [q for _, q in log[:2 * N]] == ["q0"] * N + ["q1"] * N and log[2 * N][1] == "q2"   # questions continue through warm-up


def test_a_failing_call_is_counted_as_an_error_and_the_probe_carries_on():
    clock, log = Clock(), []
    arms = all_arms(clock, log)

    def broken(question):
        raise RuntimeError("boom")
    arms["lc_plain"] = broken
    results = probe.run_probe(arms, QS, rounds=3, warmup=0, clock=clock)
    assert results["lc_plain"]["errors"] == 3 and results["lc_plain"]["ms"] == []
    assert len(results["plain"]["ms"]) == 3 and results["plain"]["errors"] == 0


def test_errors_during_warmup_are_not_counted():
    clock, log = Clock(), []
    arms = all_arms(clock, log)
    calls = []

    def fails_once(question):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("cold start")
        return {"prompt_tokens": 1, "completion_tokens": 1}
    arms["plain"] = fails_once
    results = probe.run_probe(arms, QS, rounds=2, warmup=1, clock=clock)
    assert results["plain"]["errors"] == 0 and len(results["plain"]["ms"]) == 2


def test_a_rate_limit_stops_the_probe():
    clock, log = Clock(), []
    arms = all_arms(clock, log)

    def limited(question):
        raise ProviderError("tokens per day", 429)
    arms["lc_prompt"] = limited
    with pytest.raises(ProviderError):
        probe.run_probe(arms, QS, rounds=3, warmup=0, clock=clock)


def test_a_failed_call_adds_no_time_or_tokens_to_the_record():
    clock, log = Clock(), []
    arms = all_arms(clock, log)

    def sometimes(question):
        if question == "q1":
            raise RuntimeError("x")
        clock.advance_ms(50)
        return {"prompt_tokens": 5, "completion_tokens": 2}
    arms["plain"] = sometimes
    results = probe.run_probe(arms, QS, rounds=3, warmup=0, clock=clock)
    assert results["plain"]["ms"] == pytest.approx([50.0, 50.0]) and results["plain"]["completion"] == [2, 2]


def test_missing_token_counts_are_skipped_not_recorded_as_zero():
    clock = Clock()

    def no_tokens(question):
        clock.advance_ms(10)
        return {"prompt_tokens": None, "completion_tokens": None}
    results = probe.run_probe({"a": no_tokens}, QS, rounds=2, warmup=0, clock=clock)
    assert results["a"]["ms"] == pytest.approx([10.0, 10.0]) and results["a"]["prompt"] == [] and results["a"]["completion"] == []


# ---- summarising and reading the result -----------------------------------------------------------------------------------
def test_summary_statistics_and_milliseconds_per_token():
    calls = [(300.0, 30), (100.0, 10), (400.0, 30), (200.0, 10)]
    results = {"a": results_for([100.0, 200.0, 300.0, 400.0], prompt=[50, 70], completion=[10, 30], errors=1, calls=calls)}
    s = probe.summarise_probe(results)["a"]
    assert (s["ok"], s["errors"]) == (4, 1)
    assert s["mean_ms"] == 250.0 and s["p50_ms"] == 200.0 and s["p95_ms"] == 400.0 and s["min_ms"] == 100.0
    assert s["mean_prompt_tokens"] == 60.0 and s["mean_completion_tokens"] == 20.0
    assert s["ms_per_completion_token"] == 12.5
    assert s["calls"] == calls                                                        # in the order they happened, with tokens
    assert s["fast_calls"] == 4


def test_an_arm_with_no_successes_or_no_tokens_has_none_not_a_crash():
    s = probe.summarise_probe({"dead": results_for([], errors=4), "no_tokens": results_for([100.0])})
    assert s["dead"]["mean_ms"] is None and s["dead"]["p50_ms"] is None and s["dead"]["min_ms"] is None
    assert s["dead"]["ms_per_completion_token"] is None and s["dead"]["calls"] == []
    assert s["no_tokens"]["mean_ms"] == 100.0 and s["no_tokens"]["ms_per_completion_token"] is None


def test_each_call_is_recorded_with_its_time_and_its_completion_tokens():
    clock = Clock()
    tokens = iter([11, None, 33])

    def arm_with_tokens(question):
        clock.advance_ms(10)
        return {"prompt_tokens": 1, "completion_tokens": next(tokens)}
    results = probe.run_probe({"a": arm_with_tokens}, QS, rounds=2, warmup=1, clock=clock)     # 1 warm-up + 2 measured
    assert results["a"]["calls"] == [(pytest.approx(10.0), None), (pytest.approx(10.0), 33)]     # warm-up (11) not recorded


def test_ratios_compare_each_step_with_the_one_before_it():
    summary = {"plain": {"p50_ms": 500.0}, "lc_plain": {"p50_ms": 1000.0}, "plain2": {"p50_ms": 750.0},
               "lc_prompt": {"p50_ms": 1500.0}, "lc_tools": {"p50_ms": 4500.0}}
    r = probe.ratios(summary, "p50_ms")
    assert r["noise floor    (plain2 / plain)"] == 1.5 and r["client library (lc_plain / plain)"] == 2.0
    assert r["longer prompt  (lc_prompt / lc_plain)"] == 1.5 and r["tool schemas   (lc_tools / lc_prompt)"] == 3.0
    assert r["everything     (lc_tools / plain)"] == 9.0


def test_a_ratio_is_none_when_either_side_is_missing_or_zero():
    r = probe.ratios({"plain": {"p50_ms": None}, "lc_plain": {"p50_ms": 100.0}, "lc_prompt": {"p50_ms": 0},
                      "lc_tools": {"p50_ms": 50.0}}, "p50_ms")
    assert all(v is None for v in r.values())
    assert probe.ratios({}, "p50_ms")["everything     (lc_tools / plain)"] is None


def test_the_same_ratio_can_be_read_by_median_or_by_mean_and_they_can_disagree_wildly():
    """Regression test with a REAL probe result: plain calls were bimodal, so medians and means told different stories."""
    summary = {"plain": {"mean_ms": 2831.0, "p50_ms": 537.0}, "lc_plain": {"mean_ms": 2494.0, "p50_ms": 2569.0},
               "lc_prompt": {"mean_ms": 3863.0, "p50_ms": 4455.0}, "lc_tools": {"mean_ms": 4182.0, "p50_ms": 5475.0}}
    by_median, by_mean = probe.ratios(summary, "p50_ms"), probe.ratios(summary, "mean_ms")
    everything, library = "everything     (lc_tools / plain)", "client library (lc_plain / plain)"
    assert round(by_median[everything], 1) == 10.2 and round(by_mean[everything], 2) == 1.48
    assert round(by_median[library], 2) == 4.78 and round(by_mean[library], 2) == 0.88


def test_the_report_says_what_was_measured_and_shows_every_arm(monkeypatch, capsys):
    monkeypatch.setenv("GROQ_MODEL", "some-model")
    summary = probe.summarise_probe({n: results_for([100.0 * (i + 1)], prompt=[10], completion=[5],
                                                    calls=[(100.0 * (i + 1), 5)]) for i, n in enumerate(probe.ARMS)})
    probe.print_probe(summary, rounds=6, warmup=1)
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("Latency probe: model=some-model rounds=6 warmup=1 pause=0s date=20")
    assert "| Arm | OK | Errors | Mean | p50 | p95* | Min | Under 1 s |" in out
    assert all(f"| {n} | 1 | 0 |" in out for n in probe.ARMS)
    assert "p50 ratios (1.0 = no difference;" in out and "mean ratios (1.0 = no difference;" in out
    assert "Every call in the order it happened" in out
    assert f"everything     (lc_tools / plain): {N:.2f}x" in out and "p95 is simply the slowest call" in out
    assert "  plain: 100 (5)" in out and f"  lc_tools: {100 * N} (5)" in out   # every call with its completion tokens


def test_the_report_shows_n_a_for_missing_numbers_and_question_marks_for_missing_tokens(capsys):
    summary = probe.summarise_probe({"plain": results_for([], errors=2),
                                     "lc_plain": results_for([300.0], calls=[(300.0, None)])})
    probe.print_probe(summary, rounds=2, warmup=0)
    out = capsys.readouterr().out
    assert "| plain | 0 | 2 | n/a | n/a | n/a | n/a | 0/0 | n/a | n/a | n/a |" in out
    assert "client library (lc_plain / plain): n/a" in out
    assert "  plain: no successful calls" in out and "  lc_plain: 300 (?)" in out


def test_the_per_call_listing_is_capped_so_a_long_run_stays_readable(capsys):
    calls = [(float(i), 1) for i in range(1, 51)]
    probe.print_probe(probe.summarise_probe({"plain": results_for([c[0] for c in calls], calls=calls)}), rounds=50, warmup=0)
    line = next(row for row in capsys.readouterr().out.splitlines() if row.startswith("  plain: "))
    assert line.count("(1)") == 30 and line.startswith("  plain: 1 (1), 2 (1)")


class FakeGroqSDK:
    """Stands in for groq.Groq: counts how many clients are built and records every request sent."""
    constructed: list = []
    sent: list = []
    raises = None            # set to an exception to make every request fail

    def __init__(self, api_key=None, **kwargs):
        FakeGroqSDK.constructed.append(api_key)
        import types
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        import types
        FakeGroqSDK.sent.append(kwargs)
        if FakeGroqSDK.raises is not None:
            raise FakeGroqSDK.raises
        usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=5)
        message = types.SimpleNamespace(content="an answer")
        return types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(message=message)])


def install_fake_groq(monkeypatch):
    import groq
    FakeGroqSDK.constructed, FakeGroqSDK.sent, FakeGroqSDK.raises = [], [], None
    monkeypatch.setattr(groq, "Groq", FakeGroqSDK)
    monkeypatch.setattr(groq_client, "Groq", FakeGroqSDK)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    return FakeGroqSDK


# ---- the persistent-client arm ---------------------------------------------------------------------------------------------
@pytest.fixture
def real_arms(monkeypatch):
    pytest.importorskip("langgraph")
    pytest.importorskip("chromadb")
    import agent
    from tests.agent_fakes import ScriptedChatModel, final
    monkeypatch.setattr(agent, "make_groq_llm", lambda: ScriptedChatModel(responses=[final("ok")]))
    fake = install_fake_groq(monkeypatch)
    return fake, probe.make_arms()


def test_the_persistent_client_arm_builds_one_client_however_many_calls_it_makes(real_arms):
    fake, arms = real_arms
    assert fake.constructed == ["test-key"]                          # built once, when the arms were made
    for _ in range(3):
        arms["plain_reuse"]("What is gravity?")
    assert fake.constructed == ["test-key"] and len(fake.sent) == 3   # ...and not again for any of the three calls


def test_the_plain_path_really_does_build_a_new_client_for_every_call(real_arms):
    fake, _ = real_arms
    before = len(fake.constructed)
    for _ in range(3):
        groq_client.get_learning_assistant_response("What is gravity?", temperature=0.0)
    assert len(fake.constructed) == before + 3                        # the very thing the new arm is meant to isolate


def test_the_persistent_client_arm_sends_exactly_what_the_plain_path_sends(real_arms):
    fake, arms = real_arms
    groq_client.get_learning_assistant_response("What is gravity?", temperature=0.0)
    arms["plain_reuse"]("What is gravity?")
    plain_request, reuse_request = fake.sent
    assert reuse_request == plain_request                             # same messages, model, max_tokens, temperature, extra_body
    assert reuse_request["extra_body"] == {"reasoning_effort": "low"} and reuse_request["temperature"] == 0.0
    assert reuse_request["messages"][0] == {"role": "system", "content": groq_client.SYSTEM_PROMPT}


def test_the_persistent_client_arm_follows_the_models_reasoning_setting(real_arms, monkeypatch):
    fake, arms = real_arms
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")
    groq_client.get_learning_assistant_response("What is gravity?", temperature=0.0)
    arms["plain_reuse"]("What is gravity?")
    plain_request, reuse_request = fake.sent
    assert reuse_request == plain_request and "extra_body" not in reuse_request and reuse_request["model"] == "some/other-model"


def test_the_persistent_client_arm_lets_a_rate_limit_stop_the_probe(real_arms):
    fake, arms = real_arms
    fake.raises = ProviderError("tokens per minute", 429)
    with pytest.raises(ProviderError):
        probe.run_probe({"plain_reuse": arms["plain_reuse"]}, QS, rounds=1, warmup=0)


def test_any_other_failure_of_the_persistent_client_arm_is_counted_as_an_error(real_arms):
    fake, arms = real_arms
    fake.raises = RuntimeError("connection reset")
    results = probe.run_probe({"plain_reuse": arms["plain_reuse"]}, QS, rounds=2, warmup=0)
    assert results["plain_reuse"]["errors"] == 2 and results["plain_reuse"]["ms"] == []


# ---- the real arms, built on fakes -----------------------------------------------------------------------------------------
def test_the_real_arms_ask_the_same_question_in_different_ways(monkeypatch):
    pytest.importorskip("langgraph")
    pytest.importorskip("chromadb")
    import agent
    from tests.agent_fakes import ScriptedChatModel, final
    seen = {}

    class Recording(ScriptedChatModel):
        bound: bool = False

        def bind_tools(self, tools, **kwargs):
            seen["tools"] = [t.name for t in tools]
            return self.model_copy(update={"bound": True})        # a DISTINCT model, so using it is observable

        def _generate(self, messages, *a, **k):
            seen.setdefault("systems", []).append(messages[0].content)
            seen.setdefault("bound", []).append(self.bound)
            return super()._generate(messages, *a, **k)

    monkeypatch.setattr(agent, "make_groq_llm", lambda: Recording(responses=[final("ok", input_tokens=7, output_tokens=3)]))

    def fake_plain(question, temperature=0.7):
        tracker.record_tokens(40, 9)
        return "an answer"
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", fake_plain)
    fake_groq = install_fake_groq(monkeypatch)
    arms = probe.make_arms()
    assert list(arms) == list(probe.ARMS)
    usages = {name: fn("What is gravity?") for name, fn in arms.items()}
    assert usages["plain"] == {"prompt_tokens": 40, "completion_tokens": 9}          # taken from the token counter
    assert usages["plain2"] == usages["plain"] and arms["plain2"] is arms["plain"]   # the noise-floor arm IS the plain arm again
    assert usages["plain_reuse"] == {"prompt_tokens": 11, "completion_tokens": 5} and len(fake_groq.sent) == 1
    assert all(usages[n] == {"prompt_tokens": 7, "completion_tokens": 3} for n in ("lc_plain", "lc_prompt", "lc_tools"))
    assert seen["tools"] == ["calculator", "search_notes", "current_time"]           # the schemas really are bound
    assert seen["bound"] == [False, False, True]                                     # ...and ONLY the lc_tools arm uses them
    assert seen["systems"][0] == groq_client.SYSTEM_PROMPT                           # lc_plain: the app's own prompt
    assert seen["systems"][1] == seen["systems"][2] == agent.AGENT_SYSTEM_PROMPT     # lc_prompt and lc_tools: the agent's
    assert len(agent.AGENT_SYSTEM_PROMPT) > len(groq_client.SYSTEM_PROMPT)


def test_the_plain_arm_counts_a_swallowed_api_error_as_a_failure_not_a_fast_success(monkeypatch):
    pytest.importorskip("langgraph")
    pytest.importorskip("chromadb")
    import agent
    from tests.agent_fakes import ScriptedChatModel, final
    monkeypatch.setattr(agent, "make_groq_llm", lambda: ScriptedChatModel(responses=[final("ok")]))
    monkeypatch.setattr(groq_client, "get_learning_assistant_response", lambda q, temperature=0.7: None)
    with pytest.raises(RuntimeError, match="returned no answer"):
        probe.make_arms()["plain"]("What is gravity?")


# ---- main -------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(probe, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def fake_arms(clock=None):
    clock = clock or Clock()
    return {n: arm(clock, 10 * (i + 1), [], n) for i, n in enumerate(probe.ARMS)}


def test_a_full_run_prints_the_table_and_ratios(env, monkeypatch, capsys):
    monkeypatch.setattr(probe, "make_arms", lambda: fake_arms())
    assert probe.main(["--rounds", "2", "--warmup", "0"]) == 0
    out = capsys.readouterr().out
    assert "| Arm | OK | Errors |" in out and "p50 ratios" in out


def test_main_uses_the_rounds_and_warmup_it_was_given(env, monkeypatch):
    log = []
    clock = Clock()
    monkeypatch.setattr(probe, "make_arms", lambda: {n: arm(clock, 1, log, n) for n in probe.ARMS})
    probe.main(["--rounds", "3", "--warmup", "2"])
    assert len(log) == N * (3 + 2)


@pytest.mark.parametrize("argv", [["--rounds", "0"], ["--warmup", "-1"], ["--pause", "-1"]])
def test_nonsense_round_counts_are_exit_2(env, capsys, argv):
    assert probe.main(argv) == 2
    assert "--rounds must be at least 1" in capsys.readouterr().err


def test_a_missing_key_is_exit_2(env, monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY")
    assert probe.main([]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("error", [ImportError("No module named 'langgraph'"), RuntimeError("model download failed")])
def test_setup_problems_are_exit_2(env, monkeypatch, capsys, error):
    def cannot_build():
        raise error
    monkeypatch.setattr(probe, "make_arms", cannot_build)
    assert probe.main([]) == 2
    assert "Setup problem" in capsys.readouterr().err


def test_a_rate_limit_mid_run_is_a_clean_exit_2(env, monkeypatch, capsys):
    arms = fake_arms()

    def limited(question):
        raise ProviderError("tokens per day", 429)
    arms["lc_tools"] = limited
    monkeypatch.setattr(probe, "make_arms", lambda: arms)
    assert probe.main(["--rounds", "2", "--warmup", "0"]) == 2
    err = capsys.readouterr().err
    assert "rate limit reached" in err and "Traceback" not in err


def test_the_report_prints_ratios_by_median_and_by_mean_from_the_right_statistic(capsys):
    summary = probe.summarise_probe({
        "plain": results_for([100.0, 100.0, 700.0]),        # mean 300, median 100
        "lc_tools": results_for([200.0, 200.0, 800.0]),     # mean 400, median 200
    })
    probe.print_probe(summary, rounds=3, warmup=0)
    out = capsys.readouterr().out
    p50_section, mean_section = out.split("mean ratios (")
    assert "everything     (lc_tools / plain): 2.00x" in p50_section
    assert "everything     (lc_tools / plain): 1.33x" in mean_section


# ---- pacing and the fast-call count -----------------------------------------------------------------------------------------
def test_a_pause_follows_every_call_including_warmups_and_failures_but_is_not_timed():
    clock, log, sleeps = Clock(), [], []
    arms = all_arms(clock, log)

    def flaky(question):
        raise RuntimeError("boom")
    arms["plain2"] = flaky
    results = probe.run_probe(arms, QS, rounds=2, warmup=1, clock=clock, pause=2.5, sleep=sleeps.append)
    assert sleeps == [2.5] * (N * 3)                                          # 3 rounds x every arm, failures included
    assert results["plain"]["ms"] == pytest.approx([100.0, 100.0])           # sleeping did not advance the clock


def test_no_pause_means_no_sleeping():
    clock, log, sleeps = Clock(), [], []
    probe.run_probe(all_arms(clock, log), QS, rounds=2, warmup=0, clock=clock, pause=0.0, sleep=sleeps.append)
    assert sleeps == []


def test_the_default_sleep_is_looked_up_when_the_probe_runs(monkeypatch):
    slept = []
    monkeypatch.setattr(probe.time, "sleep", slept.append)
    clock, log = Clock(), []
    probe.run_probe({"a": arm(clock, 1, log, "a")}, QS, rounds=2, warmup=0, clock=clock, pause=1.0)
    assert slept == [1.0, 1.0]


def test_main_passes_the_pause_through_and_reports_it(env, monkeypatch, capsys):
    slept = []
    monkeypatch.setattr(probe.time, "sleep", slept.append)
    monkeypatch.setattr(probe, "make_arms", lambda: fake_arms())
    assert probe.main(["--rounds", "2", "--warmup", "0", "--pause", "1.5"]) == 0
    assert slept == [1.5] * (N * 2) and "pause=1.5s" in capsys.readouterr().out


def test_fast_calls_are_those_strictly_under_one_second():
    s = probe.summarise_probe({"a": results_for([999.0, 1000.0, 1001.0])})["a"]
    assert s["fast_calls"] == 1


def test_the_table_shows_how_many_calls_were_fast(capsys):
    probe.print_probe(probe.summarise_probe({"plain": results_for([200.0, 300.0, 5000.0])}), rounds=3, warmup=0)
    assert "| 2/3 |" in capsys.readouterr().out


def test_calls_are_listed_in_the_order_they_happened_not_sorted(capsys):
    calls = [(5000.0, 10), (200.0, 20), (3000.0, 30)]
    probe.print_probe(probe.summarise_probe({"plain": results_for([c[0] for c in calls], calls=calls)}), rounds=3, warmup=0)
    assert "  plain: 5000 (10), 200 (20), 3000 (30)" in capsys.readouterr().out


def test_the_connection_reuse_and_library_ratios_separate_the_two_possible_causes():
    summary = {"plain": {"p50_ms": 1000.0}, "plain_reuse": {"p50_ms": 500.0}, "lc_plain": {"p50_ms": 600.0}}
    r = probe.ratios(summary, "p50_ms")
    assert r["connection reuse (plain_reuse / plain)"] == 0.5          # reusing the connection halves the time
    assert r["LangChain vs SDK (lc_plain / plain_reuse)"] == 1.2       # LangChain is then 20% slower than the bare SDK
    assert r["client library (lc_plain / plain)"] == 0.6


def test_the_report_prints_the_connection_reuse_and_library_ratios(capsys):
    summary = probe.summarise_probe({"plain": results_for([1000.0]), "plain_reuse": results_for([500.0]),
                                     "lc_plain": results_for([600.0])})
    probe.print_probe(summary, rounds=1, warmup=0)
    out = capsys.readouterr().out
    assert "connection reuse (plain_reuse / plain): 0.50x" in out
    assert "LangChain vs SDK (lc_plain / plain_reuse): 1.20x" in out
