"""Server entry point — delegates to root app.py.

Required by openenv validate. All actual server logic lives in the root app.py.
"""

import sys
import os

# Ensure the root package directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
