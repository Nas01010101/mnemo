"""Regression test for B1: a dangling `data` symlink must not break the default
db path resolution (memory.py's `_resolve_default_db_path` / `_DEFAULT_DB`).

Background: `data` is, on the maintainer's machine, a symlink onto an external
SSD (dev convenience for the dataset cache). If that symlink is ever checked
out on a machine where the target doesn't exist, `Path("data/tenet.db").parent
.mkdir(parents=True, exist_ok=True)` raises FileExistsError — the dirent
exists (it's a symlink), but `is_dir()` can't confirm it because the target is
unreachable, so `exist_ok` can't save it. This would break `Tenet()`, every
CLI subcommand without --db, `import tenet.mcp_server` (module-level
`_tenet = Tenet()`), and the HTTP API's default session on a fresh clone.

No LLM, no network, no embeddings — pure path-resolution logic.
Run: python scripts/test_default_db_path.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tenet import memory as M  # noqa: E402

FAILS: list[str] = []


def check(desc: str, cond: bool, detail: str = "") -> None:
    mark = "ok" if cond else "FAIL"
    print(f"  {mark} {desc}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(desc)


def _fresh_home(monkeypatch_home: Path) -> None:
    os.environ["HOME"] = str(monkeypatch_home)


def test_dangling_symlink_falls_back() -> None:
    """The exact reported bug: `data` is a symlink to a target that doesn't exist."""
    repo_root = Path(tempfile.mkdtemp(prefix="tenet_fake_repo_"))
    fake_home = Path(tempfile.mkdtemp(prefix="tenet_fake_home_"))
    (repo_root / "data").symlink_to(repo_root / "nonexistent-external-volume")

    # Prove the underlying bug actually reproduces via the naive mkdir a fresh
    # clone would otherwise hit (this is what B1's audit found) — before
    # asserting our resolver avoids it.
    try:
        (repo_root / "data" / "tenet.db").parent.mkdir(parents=True, exist_ok=True)
        raised = False
    except FileExistsError:
        raised = True
    check("naive mkdir(parents=True, exist_ok=True) on a dangling symlink DOES raise "
          "FileExistsError (confirms the bug this test guards against)", raised)

    old_home = os.environ.get("HOME")
    try:
        os.environ["HOME"] = str(fake_home)
        resolved = M._resolve_default_db_path(repo_root=repo_root)
    finally:
        if old_home is not None:
            os.environ["HOME"] = old_home
        else:
            os.environ.pop("HOME", None)

    check("resolver falls back off the dangling data/ symlink",
          resolved == fake_home / ".tenet" / "tenet.db", str(resolved))

    # And the actual failure mode: MemoryCore() must construct without raising.
    try:
        core = M.MemoryCore(resolved)
        core.close()
        constructed = True
    except Exception as e:  # noqa: BLE001 — we want to see any exception, not just FileExistsError
        constructed = False
        print(f"    (MemoryCore() raised: {e!r})")
    check("MemoryCore() at the resolved fallback path constructs cleanly", constructed)


def test_unwritable_dir_falls_back() -> None:
    """A `data/` dir that exists but isn't writable (e.g. a mounted read-only
    volume) should also fall back rather than fail at mkdir/sqlite-open time."""
    repo_root = Path(tempfile.mkdtemp(prefix="tenet_fake_repo_ro_"))
    fake_home = Path(tempfile.mkdtemp(prefix="tenet_fake_home_ro_"))
    data_dir = repo_root / "data"
    data_dir.mkdir()
    data_dir.chmod(0o500)  # read + execute, no write
    try:
        old_home = os.environ.get("HOME")
        try:
            os.environ["HOME"] = str(fake_home)
            resolved = M._resolve_default_db_path(repo_root=repo_root)
        finally:
            if old_home is not None:
                os.environ["HOME"] = old_home
            else:
                os.environ.pop("HOME", None)
        check("resolver falls back off an unwritable data/ dir",
              resolved == fake_home / ".tenet" / "tenet.db", str(resolved))
    finally:
        data_dir.chmod(0o700)  # restore so tempdir cleanup can remove it


def test_normal_dir_unaffected() -> None:
    """A real, writable data/ dir (the common case) keeps using data/tenet.db —
    this fix must not change behavior for the working case."""
    repo_root = Path(tempfile.mkdtemp(prefix="tenet_fake_repo_ok_"))
    (repo_root / "data").mkdir()
    resolved = M._resolve_default_db_path(repo_root=repo_root)
    check("resolver keeps data/tenet.db when data/ is a normal writable dir",
          resolved == repo_root / "data" / "tenet.db", str(resolved))


def test_missing_dir_unaffected() -> None:
    """No data/ entry at all (fresh clone after untracking the symlink) — resolver
    should just point at data/tenet.db (MemoryCore's own mkdir creates it fresh,
    no conflicting dirent in the way)."""
    repo_root = Path(tempfile.mkdtemp(prefix="tenet_fake_repo_missing_"))
    resolved = M._resolve_default_db_path(repo_root=repo_root)
    check("resolver points at data/tenet.db when data/ doesn't exist yet",
          resolved == repo_root / "data" / "tenet.db", str(resolved))
    core = M.MemoryCore(resolved)
    core.close()
    check("MemoryCore() creates data/ fresh with no conflicting dirent", resolved.parent.is_dir())


def test_env_override_wins() -> None:
    """TENET_DB_PATH must still win over any data/ inspection."""
    repo_root = Path(tempfile.mkdtemp(prefix="tenet_fake_repo_env_"))
    (repo_root / "data").symlink_to(repo_root / "nonexistent-external-volume")
    override = Path(tempfile.mkdtemp()) / "explicit.db"
    old = os.environ.get("TENET_DB_PATH")
    try:
        os.environ["TENET_DB_PATH"] = str(override)
        resolved = M._resolve_default_db_path(repo_root=repo_root)
    finally:
        if old is not None:
            os.environ["TENET_DB_PATH"] = old
        else:
            os.environ.pop("TENET_DB_PATH", None)
    check("TENET_DB_PATH override still wins over data/ inspection", resolved == override)


if __name__ == "__main__":
    test_dangling_symlink_falls_back()
    test_unwritable_dir_falls_back()
    test_normal_dir_unaffected()
    test_missing_dir_unaffected()
    test_env_override_wins()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        raise SystemExit(1)
    print("\nALL PASS")
