import requests


def main() -> None:
    api_url = "http://127.0.0.1:8000/extract"
    target_url = "http://127.0.0.1:9000/sample.html"
    response = requests.post(api_url, json={"url": target_url}, timeout=20)
    response.raise_for_status()
    data = response.json()
    text = data["text"]
    title = data["title"]

    assert "功能简介" in text, "正文中未提取出中文标题"
    assert "静态站点示例页面" in title, "title 中文解析失败"
    print("VERIFY_OK")
    print(f"title={title}")
    print(f"text_prefix={text[:40]}")


if __name__ == "__main__":
    main()
