"""Verify the Qwen Cloud key + endpoint work. Run: python scripts/smoke_test.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402


def main() -> int:
    print(f"Base URL: {config.QWEN_BASE_URL}")
    print(f"Model:    {config.QWEN_MODEL}")
    try:
        client = config.qwen_client()
    except RuntimeError as e:
        print(f"\n[BLOCKED] {e}")
        return 2
    try:
        resp = client.chat.completions.create(
            model=config.QWEN_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=16,
        )
        print(f"\n[OK] Qwen replied: {resp.choices[0].message.content!r}")
        print(f"     tokens: {resp.usage}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"\n[FAIL] API call errored: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
