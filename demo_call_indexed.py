import json

import requests


def main() -> None:
    api_url = "http://127.0.0.1:8000/extract_by_index"
    payload = {
        "urls": [
            "http://127.0.0.1:9000/sample.html?i=1",
            "http://127.0.0.1:9000/sample.html?i=2",
            "http://127.0.0.1:9000/sample.html?i=3",
            "http://127.0.0.1:9000/sample.html?i=4",
            "http://127.0.0.1:9000/sample.html?i=5",
            "http://127.0.0.1:9000/sample.html?i=6",
            "http://127.0.0.1:9000/sample.html?i=7",
        ],
        "index": 3,
    }

    response = requests.post(api_url, json=payload, timeout=20)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
