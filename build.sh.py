import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "platform" / "frontend" / "dist"


def build_frontend():
    if not (DIST / "index.html").exists():
        print("building frontend...")
        subprocess.run("npm install && npm run build", shell=True,
                       cwd=ROOT / "platform" / "frontend", check=True)
    # Render/Heroku expect a static dir at repo root for static sites
    out = ROOT / "static"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(DIST, out)
    print(f"published SPA -> {out}")


if __name__ == "__main__":
    build_frontend()
