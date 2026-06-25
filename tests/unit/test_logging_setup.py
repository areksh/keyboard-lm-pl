import argparse
import logging

import pytest

from pl_keyboard import logging_setup as ls


def test_levels_run_from_none_to_dev_least_to_most_verbose():
    # The ladder is ordered least -> most verbose; numeric thresholds must be
    # strictly decreasing (higher threshold = quieter logger).
    thresholds = [ls.LEVELS[name] for name in ls.LEVEL_NAMES]
    assert ls.LEVEL_NAMES[0] == "none"
    assert ls.LEVEL_NAMES[-1] == "dev"
    assert thresholds == sorted(thresholds, reverse=True)


def test_none_silences_everything_including_critical():
    assert ls.LEVELS["none"] > logging.CRITICAL


def test_dev_is_more_verbose_than_debug():
    # "debug just under dev": DEV emits strictly more than DEBUG.
    assert ls.LEVELS["dev"] < ls.LEVELS["debug"] == logging.DEBUG
    assert logging.getLevelName(ls.DEV) == "DEV"


def test_default_level_is_info():
    assert ls.DEFAULT_LEVEL == "info"
    assert ls.LEVELS["info"] == logging.INFO


def test_add_argument_registers_loglevel_with_choices_and_default():
    parser = argparse.ArgumentParser()
    ls.add_argument(parser)
    args = parser.parse_args([])
    assert args.loglevel == "info"
    args = parser.parse_args(["--loglevel", "dev"])
    assert args.loglevel == "dev"
    with pytest.raises(SystemExit):
        parser.parse_args(["--loglevel", "bogus"])


def test_configure_sets_level_and_installs_single_stderr_handler():
    logger = ls.configure("debug")
    assert logger.name == ls.LOGGER_NAME
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert logger.handlers[0].stream is __import__("sys").stderr


def test_configure_is_idempotent_and_does_not_stack_handlers():
    ls.configure("info")
    logger = ls.configure("error")
    assert len(logger.handlers) == 1  # cleared, not appended
    assert logger.level == logging.ERROR


def test_configure_dev_enables_the_custom_dev_level():
    logger = ls.configure("dev")
    assert logger.isEnabledFor(ls.DEV)
    assert logger.level == ls.DEV
