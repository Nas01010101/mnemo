"""tenet — unified CLI: chat / remember / recall / stats / sweep / serve-mcp / serve-api.

Thin terminal skin over the same Tenet() core the library, MCP server, and HTTP API
all share. `rich` is optional (the `cli` extra) — everything degrades to plain print
if it isn't installed, so a bare `pip install tenet-memory` still gets a working CLI.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _console = Console()
    _RICH = True
except ImportError:
    _console = None
    _RICH = False


def _out(msg: str = "", *, style: str | None = None) -> None:
    # markup=False: msg is arbitrary/dynamic content (LLM replies, memory text) — Rich's
    # markup parser silently EATS anything that looks like a tag, e.g. "[mcp]" vanishes.
    if _RICH:
        _console.print(msg, style=style, markup=False)
    else:
        print(msg)


def _err(msg: str) -> None:
    if _RICH:
        _console.print(msg, style="red", markup=False)
    else:
        print(msg, file=sys.stderr)


def _dim(msg: str) -> None:
    _out(msg, style="dim")


def _banner() -> None:
    from . import config
    model = config.chat_model(config.QWEN_MODEL)
    _dim(f"tenet · llm={config.LLM_PROVIDER}/{model} · embed={config.EMBED_PROVIDER}")


def _open(db: str | None):
    from .core import Tenet
    return Tenet(db) if db else Tenet()


def _friendly(e: Exception) -> str:
    from . import config
    if isinstance(e, config.ProviderError):
        return (f"memory write failed: {e.reason}\n"
                f"tip: recall/stats/doubts work offline; learn/chat need a working "
                f"{e.provider} API key (see README — 'which features need a key').")
    msg = str(e)
    if "Missing/placeholder secret" in msg:
        return f"{msg}\ntip: set EMBED_PROVIDER=local to run keyless (no API key needed)."
    return f"error: {msg}"


def _parse_as_of(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        _err(f"invalid --as-of date {s!r} (expected ISO format, e.g. 2026-01-01T12:00:00)")
        raise SystemExit(1)


def _status_of(m) -> str:
    if m.expired_at is not None:
        return "superseded"
    return "pinned" if m.pinned else "current"


# ---- subcommands -----------------------------------------------------------

def cmd_remember(args) -> int:
    m = _open(args.db)
    ids = m.ingest(args.text)
    if ids:
        _out(f"remembered {len(ids)} fact(s)", style="green")
    else:
        _dim("no durable fact found")
    m.close()
    return 0


def cmd_recall(args) -> int:
    m = _open(args.db)
    hits = m.recall(args.query, k=args.k, as_of=_parse_as_of(args.as_of))
    m.close()
    if not hits:
        _dim("(no matching memories)")
        return 0
    if _RICH:
        table = Table(show_header=True, header_style="bold")
        table.add_column("text", overflow="fold")
        table.add_column("key", style="dim")
        table.add_column("valid_at", style="dim")
        table.add_column("status")
        for h in hits:
            # Text(..., no markup parsing): memory content is arbitrary/LLM-produced
            # and must render verbatim, not be interpreted as Rich markup.
            table.add_row(Text(h.text), Text(h.key or "-"),
                          datetime.fromtimestamp(h.valid_at).strftime("%Y-%m-%d %H:%M"),
                          _status_of(h))
        _console.print(table)
    else:
        for h in hits:
            print(f"[{h.score:.2f}] {h.text}  (key={h.key or '-'}, {_status_of(h)})")
    return 0


def cmd_navigate(args) -> int:
    """Adaptive multi-hop recall — same output table as `recall`, plus a trace line."""
    m = _open(args.db)
    mems, trace = m.navigate(args.query, k=args.k, max_hops=args.max_hops, tau_gain=args.tau_gain)
    m.close()
    if not mems:
        _dim("(no matching memories)")
        return 0
    if _RICH:
        table = Table(show_header=True, header_style="bold")
        table.add_column("text", overflow="fold")
        table.add_column("key", style="dim")
        table.add_column("valid_at", style="dim")
        table.add_column("status")
        for h in mems:
            table.add_row(Text(h.text), Text(h.key or "-"),
                          datetime.fromtimestamp(h.valid_at).strftime("%Y-%m-%d %H:%M"),
                          _status_of(h))
        _console.print(table)
    else:
        for h in mems:
            print(f"[{h.score:.2f}] {h.text}  (key={h.key or '-'}, {_status_of(h)})")
    hops = trace[-1]["hop"]
    _dim(f"navigated {hops} hop(s) — {trace}")
    return 0


def cmd_stats(args) -> int:
    m = _open(args.db)
    st = m.stats()
    m.close()
    if _RICH:
        table = Table(show_header=True, header_style="bold")
        table.add_column("current"); table.add_column("superseded"); table.add_column("archived")
        table.add_row(str(st["current"]), str(st["superseded"]), str(st["archived"]))
        _console.print(table)
    else:
        print(f"current={st['current']} superseded={st['superseded']} archived={st['archived']}")
    return 0


def cmd_doubts(args) -> int:
    """Facts the learned dynamics model doubts — worth re-verifying with the user."""
    m = _open(args.db)
    doubts = m.core.uncertain_facts(threshold=args.threshold)
    m.close()
    if not doubts:
        _out(f"no doubted facts (all current facts above P(valid) ≥ {args.threshold})")
        return 0
    if _RICH:
        table = Table(show_header=True, header_style="bold",
                      title="doubted beliefs (learned fact dynamics)")
        for col in ("key", "text", "P(valid)", "age (d)", "typical lifetime (d)"):
            table.add_column(col)
        for d in doubts:
            life = d["expected_lifetime_days"]
            table.add_row(Text(d["key"]), Text(d["text"]), f"{d['p_valid']:.2f}",
                          f"{d['age_days']:.0f}", "—" if life is None else f"{life:.0f}")
        _console.print(table)
    else:
        for d in doubts:
            print(f"P={d['p_valid']:.2f}  {d['key']}: {d['text']}  (age {d['age_days']:.0f}d)")
    return 0


def cmd_sweep(args) -> int:
    m = _open(args.db)
    n = m.forget_sweep()
    st = m.stats()
    m.close()
    _out(f"archived {n} stale memories · current={st['current']} "
         f"superseded={st['superseded']} archived={st['archived']}")
    return 0


def cmd_chat(args) -> int:
    from .agent import MemoryAgent
    agent = MemoryAgent(args.db) if args.db else MemoryAgent()
    _banner()
    _dim(f"currently holding: {agent.stats()}  (Ctrl-C to exit)")
    while True:
        try:
            msg = (_console.input("[bold cyan]you[/bold cyan] › ") if _RICH
                   else input("you › ")).strip()
        except (EOFError, KeyboardInterrupt):
            _dim("\nbye — I'll remember this next time.")
            return 0
        if not msg:
            continue
        if msg.startswith("/history "):
            for t in agent.recall_history(msg[9:]):
                _out(f"   • {t}")
            continue
        try:
            out = agent.respond(msg)
        except RuntimeError as e:
            _err(_friendly(e))
            continue
        reply = out["reply"] or "(no response from the model — check LLM_PROVIDER / API key / quota)"
        _out(f"tenet › {reply}", style="bold green" if _RICH else None)
        parts = []
        if out["learned"]:
            parts.append(f"+{out['learned']} fact" + ("s" if out["learned"] != 1 else ""))
        if out.get("superseded"):
            parts.append(f"{out['superseded']} superseded")
        if parts:
            _dim(f"   [{' · '.join(parts)}]")


def cmd_serve_mcp(args) -> int:
    from .mcp_server import main as mcp_main
    mcp_main()
    return 0


def cmd_serve_api(args) -> int:
    import uvicorn
    host = "localhost" if args.host in ("0.0.0.0", "") else args.host
    _dim(f"belief-state demo → http://{host}:{args.port}/  ·  API docs → http://{host}:{args.port}/docs")
    uvicorn.run("tenet.api:app", host=args.host, port=args.port)
    return 0


# ---- bench (dispatcher lives in bench_cli.py; thin wrappers here) -----------

def cmd_bench(args) -> int:
    from . import bench_cli
    action = getattr(args, "bench_action", None)
    if action == "list":
        return bench_cli.cmd_bench_list(args, _out, _err)
    if action == "results":
        return bench_cli.cmd_bench_results(args, _out, _err)
    if action == "run":
        return bench_cli.cmd_bench_run(args, getattr(args, "extra", []), _out, _err)
    _err("usage: tenet bench {list|run|results}")
    return 1


# ---- argument parsing -------------------------------------------------------

def _add_db(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=None, help="sqlite db path (default: data/tenet.db)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tenet", description="Self-managing bi-temporal memory for LLM agents.")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("chat", help="interactive assistant with long-term memory")
    _add_db(s); s.set_defaults(func=cmd_chat)

    s = sub.add_parser("remember", help="store a fact directly (distilled + supersession)")
    s.add_argument("text"); _add_db(s); s.set_defaults(func=cmd_remember)

    s = sub.add_parser("recall", help="query memory")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5, help="max results (default: 5)")
    s.add_argument("--as-of", default=None, help="ISO date/time — time-travel recall")
    _add_db(s); s.set_defaults(func=cmd_recall)

    s = sub.add_parser("navigate", help="adaptive multi-hop recall (LLM-free, stops when new evidence saturates)")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=10, help="base top-k width per hop (default: 10)")
    s.add_argument("--max-hops", type=int, default=4, help="hard depth budget (default: 4)")
    s.add_argument("--tau-gain", type=float, default=0.15,
                   help="relevance-gain floor a hop's best new item must clear (default: 0.15)")
    _add_db(s); s.set_defaults(func=cmd_navigate)

    s = sub.add_parser("stats", help="store counts (current/superseded/archived)")
    _add_db(s); s.set_defaults(func=cmd_stats)

    s = sub.add_parser("doubts", help="facts the learned staleness model doubts (confidence hints)")
    s.add_argument("--threshold", type=float, default=0.5, help="flag facts with P(valid) below this")
    _add_db(s); s.set_defaults(func=cmd_doubts)

    s = sub.add_parser("sweep", help="run the forgetting sweep")
    _add_db(s); s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("serve-mcp", help="run the MCP server (stdio)")
    s.set_defaults(func=cmd_serve_mcp)

    s = sub.add_parser("serve-api", help="run the HTTP API (uvicorn)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve_api)

    _build_bench_parser(sub)

    return p


def _build_bench_parser(sub) -> None:
    """`tenet bench {list,run,results}` — thin dispatcher over scripts/bench_*.py."""
    b = sub.add_parser("bench", help="run/enumerate the reproducible paper benchmarks")
    bsub = b.add_subparsers(dest="bench_action")
    b.set_defaults(func=cmd_bench)

    bsub.add_parser("list", help="enumerate available benchmarks + what each reproduces")

    r = bsub.add_parser("run", help="run a benchmark via its source-of-truth script")
    r.add_argument("name", help="benchmark id (see: tenet bench list)")
    r.add_argument("--provider", choices=["qwen", "local", "ollama", "openrouter"],
                   default=None, help="env preset: local/ollama keyless embeddings + reader")
    r.add_argument("--env", action="append", metavar="KEY=VAL",
                   help="extra env override (repeatable), e.g. --env OLLAMA_MODEL=qwen2.5:3b")
    r.add_argument("--qpc", type=int, default=None, help="questions per cell (factcon/mab-ar)")
    r.add_argument("--cells", default=None, help="comma list of cells (factcon/mab-ar)")
    r.add_argument("--k", type=int, default=None, help="retrieval budget top-k")
    r.add_argument("--seed", type=int, default=None, help="random seed (lme-recall)")
    r.add_argument("--principals", type=int, default=None, help="principals (churn/knowledge-update)")
    r.add_argument("--dry-run", action="store_true", help="print the exact command+env, don't run")
    # Any flag not declared above (e.g. --updates, --keys, --hops-mh) is captured by
    # parse_known_args in main() and forwarded verbatim to the script.

    res = bsub.add_parser("results", help="pretty table of past runs (data/bench_runs.jsonl)")
    res.add_argument("--limit", type=int, default=20, help="most-recent N runs (default 20)")


def main(argv=None) -> int:
    parser = _build_parser()
    # parse_known_args so `bench run <name>` can forward script-specific flags
    # (--updates, --keys, --hops-mh, …) verbatim without declaring every one here.
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        if getattr(args, "command", None) == "bench" and getattr(args, "bench_action", None) == "run":
            args.extra = unknown
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args) or 0
    except RuntimeError as e:
        _err(_friendly(e))
        return 1
    except ImportError as e:
        _err(f"missing dependency: {e}. Install the matching extra, "
             f"e.g. pip install 'tenet-memory[mcp]' or '[api]'.")
        return 1
    except KeyboardInterrupt:
        _dim("\ninterrupted")
        return 130
    except Exception as e:  # noqa: BLE001 — CLI boundary: never show a raw traceback
        _err(f"error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
