from __future__ import annotations

import logging
import sys
from typing import TextIO

#will need some modification after v1.0.0 release
def get_logger(
    name: str,
    level: int = logging.INFO,
    stream: TextIO = sys.stdout,
) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name.
        level: Logging level.
        stream: Output stream.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def log_step(
    logger: logging.Logger,
    step: int,
    loss: float,
    extras: dict | None = None,
) -> None:
    """Log a training step.

    Args:
        logger: Logger instance.
        step: Step number.
        loss: Loss value.
        extras: Additional metrics to log.
    """
    msg = f"Step {step}: loss={loss:.4f}"
    if extras:
        extra_str = ", ".join(f"{k}={v:.4f}" for k, v in extras.items())
        msg = f"{msg}, {extra_str}"
    logger.info(msg)

