"""Learning from the sessions Claude Code kept from before the plugin was installed
(`plugin.history`): the first session start in a repository folds that repository's earlier
sessions -- micro-skills, tool pruning and the loop guard's envelope all learn from them -- and
from then on the plugin learns from the sessions it sees, as before."""
import json
import os
import time

import pytest

from tokeneconomizer import trace
from tokeneconomizer.plugin import history, hosts
from tokeneconomizer.plugin.context import LiveContext

from test_live_hooks import (call, env, repo, run_session, store_for,  # noqa: F401 (fixtures)
                             trap_session, write_session)

CLAIM = "`pytest …` fails here with command not found; `python -m pytest …` works."


@pytest.fixture
def claude_home(env, tmp_path):
    home = tmp_path / "claude-home"
    env["CLAUDE_CONFIG_DIR"] = str(home)
    return home


def kept(home, cwd, sid, make, age_s=3600):
    """A session log as Claude Code keeps it: under the directory it started in, every entry
    naming that directory, last written `age_s` ago."""
    proj = home / "projects" / "".join(c if c.isalnum() else "-" for c in str(cwd))
    proj.mkdir(parents=True, exist_ok=True)
    path = str(proj / f"{sid}.jsonl")
    make(path)
    lines = [dict(json.loads(ln), cwd=str(cwd), sessionId=sid)
             for ln in open(path, encoding="utf-8") if ln.strip()]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines) + "\n")
    t = time.time() - age_s
    os.utime(path, (t, t))
    return path


def start(env, repo, sid, tmp_path, source="startup"):
    return call(env, {"session_id": sid, "transcript_path": str(tmp_path / f"{sid}.jsonl"),
                      "cwd": repo, "hook_event_name": "SessionStart", "source": source})


def test_the_first_session_learns_from_the_sessions_kept_from_before(env, repo, tmp_path,
                                                                    claude_home):
    for i in range(5):
        kept(claude_home, repo, f"old{i}", lambda p, i=i: trap_session(p, repo, i),
             age_s=3600 * (10 - i))
    first = start(env, repo, "new1", tmp_path)
    assert "additionalContext" not in first.get("hookSpecificOutput", {})   # learned after it
    st = store_for(env, repo)
    assert st.read(history.STATE)["sessions"] == 5
    ledger = st.read_lines("ledger.jsonl")
    assert [r["session_id"] for r in ledger] == [f"old{i}" for i in range(5)]   # oldest first
    assert all(r["source"] == "history" for r in ledger)

    second = start(env, repo, "new2", tmp_path)
    assert CLAIM in second["hookSpecificOutput"]["additionalContext"]
    assert "learned from 5 earlier claude session(s)" in second["systemMessage"]
    third = start(env, repo, "new3", tmp_path)
    assert "earlier" not in third.get("systemMessage", "")                   # said once
    assert len(st.read_lines("ledger.jsonl")) == 5                           # read once


def test_learning_goes_on_from_the_sessions_the_plugin_sees(env, repo, tmp_path, claude_home):
    for i in range(2):
        kept(claude_home, repo, f"old{i}", lambda p, i=i: trap_session(p, repo, i))
    for i in range(3):
        run_session(env, repo, f"s{i}", trap_session(str(tmp_path / f"s{i}.jsonl"), repo, i))
    st = store_for(env, repo)
    assert [r.get("source") for r in st.read_lines("ledger.jsonl")] == [
        "history", "history", None, None, None]
    assert st.read("injection.json")["sessions"] == 5
    assert CLAIM in start(env, repo, "next", tmp_path)["hookSpecificOutput"]["additionalContext"]


def test_only_this_repositorys_finished_sessions_are_read(env, repo, tmp_path, claude_home):
    other = tmp_path / "other"
    (other / ".git").mkdir(parents=True)
    (other / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    kept(claude_home, other, "elsewhere", lambda p: trap_session(p, str(other), 0))
    kept(claude_home, tmp_path / "gone", "deleted", lambda p: trap_session(p, repo, 1))
    kept(claude_home, repo, "running", lambda p: trap_session(p, repo, 2), age_s=5)
    os.makedirs(os.path.join(repo, "pkg"), exist_ok=True)
    kept(claude_home, os.path.join(repo, "pkg"), "subdir", lambda p: trap_session(p, repo, 3))
    kept(claude_home, repo, "mine", lambda p: trap_session(p, repo, 4))
    start(env, repo, "new", tmp_path)
    assert sorted(store_for(env, repo).read("folded.json")) == ["mine", "subdir"]


def test_a_session_the_hooks_saw_is_folded_by_them_not_from_history(env, repo, tmp_path,
                                                                   claude_home):
    path = kept(claude_home, repo, "seen", lambda p: trap_session(p, repo, 0))
    run_session(env, repo, "seen", path)
    recs = store_for(env, repo).read_lines("ledger.jsonl")
    assert [(r["session_id"], r.get("source")) for r in recs] == [("seen", None)]


def test_history_off_reads_nothing(env, repo, tmp_path, claude_home):
    env["TE_HISTORY"] = "off"
    kept(claude_home, repo, "old", lambda p: trap_session(p, repo, 0))
    start(env, repo, "new", tmp_path)
    assert store_for(env, repo).read_lines("ledger.jsonl") == []
    assert store_for(env, repo).read(history.STATE) is None


def test_the_loop_guard_learns_its_envelope_from_turns_that_ended_well(env, repo, tmp_path,
                                                                      claude_home):
    def session(path):
        ok = ("Bash", {"command": "ls"}, "a b", False)
        write_session(path, [("one", [ok, ok]), ("two", [ok, ok, ok]), ("three", [ok])])
        with open(path, "a", encoding="utf-8") as fh:     # the person interrupted the third
            fh.write(json.dumps({"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "[Request interrupted by user]"}]}}) + "\n")
    kept(claude_home, repo, "old", session)
    start(env, repo, "new", tmp_path)
    assert store_for(env, repo).read("envelope.json")["resolved_turns"] == [2, 3]
    assert store_for(env, repo).read(history.STATE)["turns"] == 2


def test_a_run_that_died_is_started_again_later(env, repo, tmp_path, claude_home):
    kept(claude_home, repo, "old", lambda p: trap_session(p, repo, 0))
    ctx = LiveContext.build(hosts.ClaudeCode(), repo, env)
    ctx.store.write(history.STATE, {"started": time.time()})
    assert not history.start(ctx)
    assert history.start(ctx, now=time.time() + history.RETRY_S + 1)
    assert ctx.store.read(history.STATE)["sessions"] == 1


def test_te_history_folds_what_is_still_unfolded(env, repo, tmp_path, claude_home, monkeypatch,
                                                 capsys):
    from tokeneconomizer.plugin import cli
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    kept(claude_home, repo, "old", lambda p: trap_session(p, repo, 0))
    assert cli.main(["--host", "claude", "--cwd", repo, "history"]) == 0
    assert "folded 1 earlier session(s)" in capsys.readouterr().out
    assert cli.main(["--host", "claude", "--cwd", repo, "history"]) == 0
    assert "folded 0 earlier session(s) of 0 found" in capsys.readouterr().out


def test_prompt_turns_split_a_claude_log_by_the_persons_prompts(tmp_path):
    path = str(tmp_path / "t.jsonl")
    write_session(path, [("a", [("Read", {"file_path": "x"}, "x", False)]),
                         ("b", [("Bash", {"command": "false"}, "Exit code 1", True)])])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "assistant", "isSidechain": True, "message": {"content": [
            {"type": "tool_use", "id": "side", "name": "Grep", "input": {}}]}}) + "\n")
    assert trace.claude_prompt_turns(path) == [
        ([("Read", {"file_path": "x"}, "x", True)], False),
        ([("Bash", {"command": "false"}, "Exit code 1", False)], False)]


def test_a_deny_list_learned_from_history_waits_for_its_sessions_to_go_quiet(
        env, repo, tmp_path, claude_home):
    """Folded from history there is no session record, only the log: a deny list waits on it as
    on any session's, since that conversation could be resumed with a warm prompt cache."""
    from tokeneconomizer.plugin import fold
    env["TE_TOOLPRUNE_QUIET_MINUTES"] = "60"
    log = kept(claude_home, repo, "old", lambda p: trap_session(p, repo, 0), age_s=900)
    ctx = LiveContext.build(hosts.ClaudeCode(), repo, env)
    history.run(ctx)
    assert abs(fold.last_activity(ctx) - os.path.getmtime(log)) < 1


def test_a_deny_list_learned_from_history_waits_for_apply(env, repo, tmp_path, claude_home,
                                                          monkeypatch, capsys):
    """Under `auto`, what the history proposes is announced and written only on `apply`; the
    plugin goes on writing what it learns from the sessions it sees by itself."""
    from test_live_hooks import _catalog, _deny
    from tokeneconomizer.plugin import cli
    _catalog(env, ["Bash", "Read", "CronCreate", "Workflow"])
    for i in range(5):
        kept(claude_home, repo, f"old{i}", lambda p: write_session(
            p, [("p", [("Bash", {"command": "ls"}, "a", False)])]))
    start(env, repo, "new1", tmp_path)
    assert _deny(repo) == []
    second = start(env, repo, "new2", tmp_path)
    assert "nothing changes until you run" in second["systemMessage"]
    assert "CronCreate" in second["systemMessage"]

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--host", "claude", "--cwd", repo, "toolprune"]) == 0
    assert "written only when you run `toolprune apply`" in capsys.readouterr().out
    assert cli.main(["--host", "claude", "--cwd", repo, "toolprune", "apply"]) == 0
    assert _deny(repo) == ["CronCreate", "Workflow"]



def test_a_deny_list_the_plugins_own_sessions_complete_is_written_by_itself(
        env, repo, tmp_path, claude_home):
    """History too short to propose anything still counts: the sessions the plugin sees next
    complete the evidence, and `auto` writes the list as it always has."""
    from test_live_hooks import _catalog, _deny
    _catalog(env, ["Bash", "Read", "CronCreate"])
    for i in range(3):
        kept(claude_home, repo, f"old{i}", lambda p: write_session(
            p, [("p", [("Bash", {"command": "ls"}, "a", False)])]))
    for i in range(2):
        run_session(env, repo, f"s{i}", write_session(str(tmp_path / f"s{i}.jsonl"), [
            ("p", [("Bash", {"command": "ls"}, "a", False)])]))
    assert _deny(repo) == ["CronCreate"]
