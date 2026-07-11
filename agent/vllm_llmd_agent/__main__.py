"""CLI entrypoint: ``vllm-llmd-agent``."""

from __future__ import annotations

import logging
import sys

from .agent import Agent
from .config import Config


def main(argv: list[str] | None = None) -> int:
    cfg = Config.from_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    Agent(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
