"""``python -m meridian.audit verify`` walks the hash chain."""

import argparse
import sys
from pathlib import Path

from meridian.audit.store import broken, install_from_env, sql_store, verify_sql, verify_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meridian.audit")
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--dir", default="data/audit", dest="directory")
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    kind = install_from_env(directory)
    if kind == "postgres":
        store = sql_store()
        if store is None:
            print("audit store missing", file=sys.stderr)
            return 1
        lines = verify_sql(store)
        print("audit store: postgres")
    else:
        lines = verify_tree(directory)
        print("audit store: jsonl")
    for line in lines:
        print(line)
    return 1 if broken(lines) else 0


if __name__ == "__main__":
    raise SystemExit(main())
