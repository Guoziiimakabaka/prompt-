import json
import sys

import requests


def main() -> None:
    url = "http://127.0.0.1:8000/extract"
    target = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9000/sample.html"
    payload = {"url": target}
    response = requests.post(url, json=payload, timeout=20)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
