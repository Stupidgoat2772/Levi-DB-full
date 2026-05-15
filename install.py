#!/usr/bin/env python3
"""install.py — Levi-DB setup script.

Downloads source data, installs Python dependencies, and optionally
builds the vector database. Works on Linux, macOS, and Windows.

Usage:
    python install.py              # full install (data + deps + build)
    python install.py --deps-only  # just install Python packages
    python install.py --data-only  # just download source texts
    python install.py --no-build   # install everything but skip DB build
    python install.py --help
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LEVI_DATA_DIR", SCRIPT_DIR / "data"))

# Source repositories
DATA_REPOS = {
    "morphhb": {
        "url": "https://github.com/openscriptures/morphhb.git",
        "desc": "Hebrew OT (morphhb)",
        "license": "CC BY 4.0",
    },
    "sblgnt": {
        "url": "https://github.com/morphgnt/sblgnt.git",
        "desc": "Greek NT (SBLGNT/morphgnt)",
        "license": "CC BY-SA",
    },
    "strongs": {
        "url": "https://github.com/openscriptures/strongs.git",
        "desc": "Strong's dictionaries",
        "license": "Public domain",
    },
    "sdbh": {
        "url": "https://github.com/pthu/sdbh.git",
        "desc": "UBS semantic domains (SDBH)",
        "license": "CC BY-SA 4.0",
    },
}

PIP_PACKAGES = ["sentence-transformers", "numpy", "ollama"]

OLLAMA_MODELS = ["nomic-embed-text"]

# ANSI colors (disabled on Windows cmd without VT support)
if sys.stdout.isatty() and (platform.system() != "Windows" or "WT_SESSION" in os.environ):
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
else:
    GREEN = YELLOW = RED = BOLD = RESET = ""


def heading(msg):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")


def ok(msg):
    print(f"  {GREEN}[OK]{RESET} {msg}")


def warn(msg):
    print(f"  {YELLOW}[!!]{RESET} {msg}")


def fail(msg):
    print(f"  {RED}[FAIL]{RESET} {msg}")


def run(cmd, **kwargs):
    """Run a command, return (success, output)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, **kwargs)
        return result.returncode == 0, result.stdout.strip()
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return False, "Command timed out"


# ── Checks ──────────────────────────────────────────────────────────

def check_python():
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    fail(f"Python {v.major}.{v.minor} — need 3.10+")
    return False


def check_git():
    success, out = run(["git", "--version"])
    if success:
        ok(f"git ({out})")
        return True
    fail("git not found — install it from https://git-scm.com")
    return False


def check_ollama():
    success, out = run(["ollama", "--version"])
    if success:
        ok(f"Ollama installed ({out.strip()})")
        return True
    warn("Ollama not found — install from https://ollama.ai")
    warn("You'll need it before building the DB or running queries.")
    return False


def check_ollama_running():
    success, out = run(["ollama", "list"])
    if success:
        ok("Ollama is running")
        return True
    warn("Ollama is installed but not running. Start it:")
    if platform.system() == "Darwin":
        warn("  Open the Ollama app, or: ollama serve")
    elif platform.system() == "Windows":
        warn("  Open the Ollama app from Start Menu")
    else:
        warn("  systemctl start ollama  OR  ollama serve")
    return False


def check_ollama_model(model):
    success, out = run(["ollama", "list"])
    if success and model in out:
        ok(f"Model '{model}' available")
        return True
    return False


# ── Actions ─────────────────────────────────────────────────────────

def install_deps():
    heading("Installing Python dependencies")
    pip_cmd = [sys.executable, "-m", "pip", "install"] + PIP_PACKAGES
    print(f"  Running: {' '.join(pip_cmd)}\n")
    result = subprocess.run(pip_cmd)
    if result.returncode == 0:
        ok("Python dependencies installed")
        return True
    fail("pip install failed — check output above")
    return False


def clone_data():
    heading("Downloading source texts")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in DATA_REPOS.items():
        dest = DATA_DIR / name
        if dest.exists() and any(dest.iterdir()):
            ok(f"{info['desc']} — already downloaded")
            continue

        print(f"  Cloning {info['desc']}...")
        print(f"    {info['url']}")
        print(f"    License: {info['license']}")

        success, out = run(["git", "clone", "--depth", "1", info["url"], str(dest)])
        if success:
            # Remove .git to save space — these are read-only data
            git_dir = dest / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)
            ok(f"{info['desc']} — downloaded")
        else:
            fail(f"Failed to clone {name}: {out}")
            return False

    return True


def pull_models():
    heading("Pulling Ollama models")
    for model in OLLAMA_MODELS:
        if check_ollama_model(model):
            continue
        print(f"  Pulling {model} (this may take a few minutes)...")
        result = subprocess.run(["ollama", "pull", model])
        if result.returncode == 0:
            ok(f"Model '{model}' ready")
        else:
            fail(f"Failed to pull {model}")
            return False
    return True


def build_db():
    heading("Building Levi-DB")
    build_script = SCRIPT_DIR / "build_levi_db.py"
    env = os.environ.copy()
    env["LEVI_DATA_DIR"] = str(DATA_DIR)

    print("  This will take 30-90 minutes depending on your hardware.")
    print("  Embedding ~23,000 verses with nomic-embed-text...\n")

    result = subprocess.run([sys.executable, str(build_script)], env=env)
    if result.returncode == 0:
        ok("Levi-DB built successfully")
        return True
    fail("Build failed — check output above")
    return False


def verify_data():
    """Check that all required data files exist."""
    required = [
        DATA_DIR / "morphhb" / "wlc",
        DATA_DIR / "strongs" / "hebrew" / "strongs-hebrew-dictionary.js",
        DATA_DIR / "strongs" / "greek" / "strongs-greek-dictionary.js",
        DATA_DIR / "sdbh" / "dictionaries" / "hebrew" / "JSON",
        DATA_DIR / "sdbh" / "dictionaries" / "greek" / "JSON",
        DATA_DIR / "sblgnt",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        fail("Missing data files:")
        for p in missing:
            print(f"    {p}")
        return False
    ok("All source data present")
    return True


# ── Main ────────────────────────────────────────────────────────────

def main():
    args = set(sys.argv[1:])

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    deps_only = "--deps-only" in args
    data_only = "--data-only" in args
    no_build = "--no-build" in args

    print(f"""
{BOLD}Levi-DB Installer{RESET}
Enriched Hebrew/Greek Bible vector database

  Script dir:  {SCRIPT_DIR}
  Data dir:    {DATA_DIR}
  Platform:    {platform.system()} {platform.machine()}
  Python:      {sys.version.split()[0]}
""")

    # ── Prerequisites ──
    heading("Checking prerequisites")
    py_ok = check_python()
    git_ok = check_git()
    ollama_ok = check_ollama()
    ollama_running = check_ollama_running() if ollama_ok else False

    if not py_ok:
        fail("Python 3.10+ required. Aborting.")
        sys.exit(1)

    if not git_ok and not deps_only:
        fail("git required for downloading source texts. Aborting.")
        sys.exit(1)

    # ── Deps ──
    if not data_only:
        if not install_deps():
            sys.exit(1)

    if deps_only:
        print(f"\n{GREEN}Dependencies installed. Run again without --deps-only to continue.{RESET}")
        return

    # ── Data ──
    if not clone_data():
        sys.exit(1)

    if not verify_data():
        sys.exit(1)

    if data_only:
        print(f"\n{GREEN}Data downloaded. Run again without --data-only to continue.{RESET}")
        return

    # ── Ollama models ──
    if ollama_ok and ollama_running:
        pull_models()
    else:
        warn("Skipping model pull — Ollama not available.")
        warn("Install/start Ollama, then run: ollama pull nomic-embed-text")

    # ── Build ──
    if no_build:
        print(f"\n{GREEN}Setup complete (--no-build). Build the DB when ready:{RESET}")
        print(f"  python3 build_levi_db.py")
        return

    if not ollama_running:
        warn("Can't build DB without Ollama running.")
        warn("Start Ollama, pull nomic-embed-text, then run:")
        warn("  python3 build_levi_db.py")
        return

    if not check_ollama_model("nomic-embed-text"):
        warn("nomic-embed-text not found. Pull it first:")
        warn("  ollama pull nomic-embed-text")
        return

    build_db()

    # ── Done ──
    heading("Setup complete")
    print(f"""  {GREEN}Levi-DB is ready.{RESET}

  Try it:
    python3 oracle.py "What does it mean that God rested on the seventh day?"
    python3 ask_levi.py "love your neighbor"

  For more options:
    python3 oracle.py --help
    python3 ask_levi.py --help
""")


if __name__ == "__main__":
    main()
