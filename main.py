"""Entry point for the AI-Powered Object Detection System.

Parses command-line arguments, applies any overrides to the global
settings, configures logging, and launches the FastAPI server via
uvicorn.

Usage:
    python main.py                          # defaults from config
    python main.py --host 0.0.0.0 --port 8080
    python main.py --model yolo11s --device cuda
    python main.py --reload                 # dev mode with auto-reload
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from app.config import ModelSize, settings


def _configure_logging(level: str) -> None:
    """Set up root logging with a consistent format.

    Args:
        level: Logging level string (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Object Detection server.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="AI-Powered Object Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=settings.api_host,
        help="Host address to bind the API server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help="Port number for the API server.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes.",
    )
    parser.add_argument(
        "--model",
        choices=["yolo11n", "yolo11s", "yolo11m"],
        default=None,
        help="YOLO model size to use.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 'cuda', 'cpu', or 'auto'.",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: apply CLI overrides and start the uvicorn server."""
    args = parse_args()

    # Apply CLI overrides to global settings before the app starts
    if args.model:
        try:
            settings.model_size = ModelSize(args.model)
        except ValueError:
            print(
                f"Error: Invalid model '{args.model}'. "
                f"Choose from: {[m.value for m in ModelSize]}",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.device:
        settings.device = args.device

    log_level = args.log_level or settings.log_level
    _configure_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  Object Detection System — Starting Up")
    logger.info("=" * 60)
    logger.info("  Host      : %s", args.host)
    logger.info("  Port      : %d", args.port)
    logger.info("  Model     : %s", settings.model_size.value)
    logger.info("  Device    : %s", settings.device or "auto")
    logger.info("  Workers   : %d", args.workers)
    logger.info("  Reload    : %s", args.reload)
    logger.info("  Log Level : %s", log_level)
    logger.info("=" * 60)

    try:
        uvicorn.run(
            "app.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=log_level.lower(),
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as exc:
        logger.error("Failed to start server: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
