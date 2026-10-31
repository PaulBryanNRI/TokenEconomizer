"""Learning from the sessions a person ran before the plugin saw them.

The hooks see only the sessions that run after the plugin is installed, so on its own the plugin
would start every repository from nothing. Claude Code keeps every session's log
(`~/.claude/projects/<encoded cwd>/<session-id>.jsonl`), and those logs are what a fold reads
anyway. So the first time the plugin runs in a repository, it folds that repository's earlier
sessions too -- all of them, oldest first -- before learning on from
the sessions it sees:

  * micro-skills and tool pruning learn from each one through the ordinary fold
    (`fold.fold_session`), which records it in the ledger as `source: "history"`;
  * the loop guard's envelope learns from each one's prompt-turns, replayed through the same
    `TurnGuard` the `PostToolUse` and `Stop` hooks drive (`replay_turns`), since the hooks that
    feed it live never ran for them.

A session belongs to the repository of the directory it started in -- Claude Code files its log
under that directory's name, and the log records it (`cwd`) -- identified as the plugin
identifies any session's repository (`paths.repo_identity`). Skipped: a session the
hooks have a record of (the fold handles it), one already folded, and one whose log was written
in the last `fold.STALE_S` seconds (it may still be running). It all runs in a detached `te
history`, started once per repository and agent by `SessionStart` (`start`); `history.json`
records that it ran. `history=off` turns it off.
"""
import os
import time

from .. import loopguard
from . import fold
from .hosts import _EXIT_RE
from .paths import repo_identity
from .repos import _transcript_cwd

STATE = "history.json"
RETRY_S = 3600          # a run that started this long ago and never finished is started again


def due(ctx, now=None):
    """Should the history be read now: on, an agent whose history is read, and never read for
    this repository (or a run that died part way)?"""
    if not ctx.config.history or ctx.config.shadow:
        return False
    if not ctx.host.reads_history:
        return False
    st = ctx.store.read(STATE)
    if not isinstance(st, dict):
        return True
    return not st.get("done") and (now or time.time()) - float(st.get("started") or 0) > RETRY_S


def start(ctx, now=None):
    """Read the repository's history in a detached `te history` (inline when TE_FOLD_SYNC=1),
    if it is due. Returns whether it started."""
    if not due(ctx, now):
        return False
    with ctx.store.locked("history", timeout=0) as got:
        if not got or not due(ctx, now):
            return False
        prior = ctx.store.read(STATE) or {}
        ctx.store.write(STATE, dict(prior, started=now or time.time()))
    if ctx.env.get("TE_FOLD_SYNC") == "1":
        run(ctx)
    else:
        ctx.spawn(["history"], "fold.log")
    return True


def candidates(ctx, now=None):
    """[(session id, log)] of this repository's earlier sessions to fold, oldest first."""
    now = now or time.time()
    folded = ctx.store.read("folded.json") or {}
    sess_dir = ctx.store.path("sessions")
    found = []
    for logs in ctx.host.history_logs(ctx.env).values():
        # One directory holds the sessions started in one directory (it is named for it), so
        # the first log that names its directory says which repository they all belong to.
        cwd = next((c for c in map(_transcript_cwd, logs) if c), None)
        if not cwd or _repo_key(cwd, ctx.env) != ctx.repo.key:
            continue
        for log in logs:
            sid = os.path.splitext(os.path.basename(log))[0]
            if sid in folded or os.path.exists(os.path.join(sess_dir, f"{sid}.json")):
                continue
            try:
                mtime = os.path.getmtime(log)
            except OSError:
                continue
            if now - mtime >= fold.STALE_S:
                found.append((mtime, sid, log))
    return [(sid, log) for _, sid, log in sorted(found)]


def _repo_key(cwd, env):
    """The state key of the repository `cwd` is in; None once the directory is gone."""
    try:
        return repo_identity(cwd, env).key if os.path.isdir(cwd) else None
    except OSError:
        return None


def replay_turns(ctx, transcript):
    """Feed a finished session's prompt-turns to the loop guard's envelope, as its `PostToolUse`
    and `Stop` hooks would have: each call folded into a `TurnGuard` under the thresholds the
    hook would have used, and a turn that ran to its end recorded (`loopguard.record_turn`,
    which keeps only the ones that ended green). Returns how many turns were recorded."""
    cfg = ctx.config
    n = 0
    for calls, interrupted in ctx.host.prompt_turns(transcript):
        if interrupted or not calls:
            continue
        tracker = loopguard.load_envelope(ctx.store, level=cfg.loopguard_aggr,
                                          overrides=cfg.loopguard_thresholds)
        thr = loopguard.thresholds(tracker, cfg.loopguard_thresholds)
        thr = dict(thr, gate_over_envelope=False,
                   gate_min_turns=max(thr["gate_min_turns"], thr["min_turns"]))
        g = loopguard.TurnGuard()
        for name, args, result, ok in calls:
            m = None if ok else _EXIT_RE.search((result or "")[:400])
            g.fold(name, args, result, int(m.group(1)) if m else None,
                   envelope=tracker.envelope, thr=thr, judge=None,
                   nudge_budget=cfg.loopguard_nudges)
        loopguard.record_turn(ctx.store, g, level=cfg.loopguard_aggr,
                              overrides=cfg.loopguard_thresholds)
        n += 1
    return n


def run(ctx, now=None):
    """Fold this repository's earlier sessions (`candidates`), and record that it was done.
    Returns {"sessions": folded, "turns": replayed, "found": candidates}."""
    found = candidates(ctx, now)
    sessions = turns = 0
    for sid, log in found:
        try:
            rec = fold.fold_session(ctx, sid, transcript_path=log, source="history")
        except Exception as e:          # one unreadable log must not stop the rest
            ctx.store.append("events.jsonl", {"kind": "history_error", "session_id": sid,
                                              "error": repr(e), "at": time.time()})
            continue
        if rec is None or rec.get("error"):
            continue
        sessions += 1
        if ctx.config.loopguard != "off":
            turns += replay_turns(ctx, log)
    summary = {"sessions": sessions, "turns": turns, "found": len(found)}
    with ctx.store.locked("history", timeout=10.0):
        prior = ctx.store.read(STATE) or {}
        ctx.store.write(STATE, dict(prior, done=time.time(),
                                    sessions=int(prior.get("sessions") or 0) + sessions,
                                    turns=int(prior.get("turns") or 0) + turns))
    return summary
