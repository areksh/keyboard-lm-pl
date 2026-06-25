"""The project's logging ladder, shared by every CLI.

A pure helper (stdlib only): it maps the ``--loglevel`` choice to a stdlib
logging threshold and configures a single stderr handler on the ``pl_keyboard``
logger. The ladder runs from ``none`` (silence) to ``dev`` — a custom level
*below* ``DEBUG`` for the most verbose, developer-only firehose.

CLIs keep their final result line on stdout (a plain ``print``); everything
diagnostic — progress, per-item detail — goes through this logger to stderr, so
``--loglevel none`` quietens the chatter without hiding the result.
"""

import logging
import sys

# A custom level beneath DEBUG(10): the most verbose rung of the ladder. Code
# logs with ``logger.log(logging_setup.DEV, ...)`` for per-step detail that even
# DEBUG would drown in.
DEV = 5
logging.addLevelName(DEV, "DEV")

# Least -> most verbose. ``none`` sits above CRITICAL so nothing is ever emitted.
LEVELS: dict[str, int] = {
    "none": logging.CRITICAL + 1,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "dev": DEV,
}
LEVEL_NAMES: list[str] = list(LEVELS)
DEFAULT_LEVEL = "info"
LOGGER_NAME = "pl_keyboard"


def add_argument(parser) -> None:
    """Register the shared ``--loglevel`` option on an argparse parser."""
    parser.add_argument(
        "--loglevel",
        choices=LEVEL_NAMES,
        default=DEFAULT_LEVEL,
        metavar="{" + "|".join(LEVEL_NAMES) + "}",
        help="Verbosity (least -> most): none < error < warning < info (default) < debug < dev.",
    )


def configure(level_name: str) -> logging.Logger:
    """Configure and return the project logger at the chosen level.

    Idempotent: re-running replaces the handler rather than stacking another, so
    repeated calls (e.g. across tests) don't multiply output.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LEVELS[level_name])
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger
