"""Start Kept: python run.py  then open http://localhost:8000 (or the phone link it prints)."""

import os
import socket
from pathlib import Path


def load_env_file():
    """Read NEBIUS_API_KEY (and other settings) from a .env file next to this one, if present.
    The .env file is listed in .gitignore, so the key never goes to GitHub."""
    f = Path(__file__).resolve().parent / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env_file()

from kept.web.app import create_app  # noqa: E402


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    ip = lan_ip()
    print("\n  Kept is running.")
    print(f"  On this computer:  http://localhost:{port}")
    if ip:
        print(f"  On your phone (same Wi-Fi):  http://{ip}:{port}")
    print("  AI: NVIDIA Nemotron on Nebius Token Factory" if os.environ.get("NEBIUS_API_KEY")
          else "  AI: off (add NEBIUS_API_KEY to a .env file to turn on NVIDIA Nemotron)")
    print("  Stop it with Ctrl+C.\n")
    create_app().run(host="0.0.0.0", port=port, debug=False)
