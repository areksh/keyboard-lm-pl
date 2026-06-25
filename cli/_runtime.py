"""Shared CLI runtime helpers: the ``--loglevel`` flag, logger setup, and a tqdm
progress wrapper.

The progress wrapper lives here (not in ``pl_keyboard``) so the pure library
imports nothing display-related; tqdm is imported lazily inside ``progress`` so
merely importing a CLI module still pulls in no third-party code.
"""

import logging

from pl_keyboard import logging_setup


def add_common_args(parser) -> None:
    """Register the options every CLI shares (currently just ``--loglevel``)."""
    logging_setup.add_argument(parser)


def configure(args) -> logging.Logger:
    """Configure the project logger from parsed args and return it."""
    return logging_setup.configure(args.loglevel)


def progress(iterable, *, desc, log, total=None, unit="it"):
    """Wrap ``iterable`` in a tqdm progress bar, shown only at INFO+ verbosity.

    The bar is suppressed when the logger is quieter than INFO (``warning`` /
    ``error`` / ``none``), so progress tracks the same threshold as other
    informational output.
    """
    from tqdm import tqdm

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        disable=not log.isEnabledFor(logging.INFO),
    )
