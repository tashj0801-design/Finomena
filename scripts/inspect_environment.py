from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finomena.environment import collect_environment


if __name__ == "__main__":
    print(json.dumps(collect_environment(), indent=2))
