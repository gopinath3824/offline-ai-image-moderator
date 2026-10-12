"""Build script for packaging Offline AI Image Moderator into a standalone Windows executable (.exe).

Uses PyInstaller to bundle CustomTkinter assets, Pillow, SQLite, and application modules.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_executable() -> int:
    """Run PyInstaller build with configured entry point and bundled assets."""
    print("=======================================================")
    print("  Building Offline AI Image Moderator Windows Executable")
    print("=======================================================\n")

    root_dir = Path(__file__).parent.resolve()
    main_py = root_dir / "main.py"

    if not main_py.exists():
        print(f"[ERROR] Entry point not found: {main_py}", file=sys.stderr)
        return 1

    # Check for PyInstaller
    try:
        import importlib
        pyinstaller = importlib.import_module("PyInstaller")
        version = getattr(pyinstaller, "__version__", "unknown")
        print(f"[OK] PyInstaller version: {version}")
    except ImportError:
        print("[ERROR] PyInstaller is not installed. Run: pip install pyinstaller", file=sys.stderr)
        return 1

    dist_dir = root_dir / "dist"
    build_dir = root_dir / "build"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=OfflineAIImageModerator",
        "--noconfirm",
        "--onedir",  # Recommended for rapid startup and large torch/transformer dependencies
        "--windowed",  # Suppress console window for desktop GUI (use --console for debug)
        "--collect-all=customtkinter",
        "--collect-all=PIL",
        "--hidden-import=sqlite3",
        "--hidden-import=config",
        "--hidden-import=ui",
        "--hidden-import=moderation",
        "--hidden-import=storage",
        "--hidden-import=utils",
        str(main_py),
    ]

    print("Executing build command:")
    print(" ".join(cmd))
    print("\nCompiling... (this may take 2-4 minutes depending on system hardware)\n")

    result = subprocess.run(cmd, cwd=str(root_dir))

    if result.returncode == 0:
        exe_path = dist_dir / "OfflineAIImageModerator" / "OfflineAIImageModerator.exe"
        print("\n=======================================================")
        print("  BUILD SUCCESSFUL!")
        print("=======================================================")
        print(f"Standalone executable located at:\n  {exe_path}\n")
        print("Distribution folder can now be zipped and distributed to Windows users.")
        print("Models can be placed in ~/.offline_image_moderator/models/ on target machines.")
        print("=======================================================\n")
        return 0
    else:
        print(f"\n[ERROR] Build failed with exit code: {result.returncode}", file=sys.stderr)
        return result.returncode


if __name__ == "__main__":
    sys.exit(build_executable())
