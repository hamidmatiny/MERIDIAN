"""``python -m meridian.bench --split dev|test|hard``."""

import argparse

from meridian.bench.harness import run_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meridian.bench")
    parser.add_argument("--split", required=True, choices=("dev", "test", "hard"))
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-key", default=None)
    args = parser.parse_args(argv)
    directory = run_split(
        args.split,
        final=args.final,
        force=args.force,
        resume=args.resume,
        only_key=args.only_key,
    )
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
