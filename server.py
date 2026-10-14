"""Server runner for Offline AI Image Moderator REST API.

Usage:
    py server.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

from __future__ import annotations

import argparse
import sys
import uvicorn

from config import APP_NAME, APP_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{APP_VERSION} - REST API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host network interface to bind (default: 127.0.0.1, use 0.0.0.0 for LAN access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code modifications (development mode)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"  {APP_NAME} v{APP_VERSION} — REST API Server")
    print("=" * 60)
    print(f"  URL:         http://{args.host}:{args.port}")
    print(f"  Docs (UI):   http://{args.host}:{args.port}/docs")
    print(f"  OpenAPI:     http://{args.host}:{args.port}/openapi.json")
    print(f"  Health:      http://{args.host}:{args.port}/health")
    print("=" * 60 + "\n")

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
