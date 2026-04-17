import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from app import extract_page_content


@dataclass
class ParsedInput:
    base_prompt: str
    repo_urls: list[str]
    scores: list[int]
    remarks: dict[int, str]


@dataclass
class RepoResult:
    repo_id: str
    url: str
    score: int
    remark: str
    ok: bool
    title: str
    text_preview: str
    text_length: int
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse input.md, fetch repo pages, and generate output.md."
    )
    parser.add_argument(
        "--input_md",
        default="input.md",
        help="Input markdown path.",
    )
    parser.add_argument(
        "--output_md",
        default="output.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--fetch_json",
        default="repo_fetch_results.json",
        help="Fetched repo debug JSON path.",
    )
    return parser.parse_args()


def _extract_json_arrays(text: str) -> list[list]:
    arrays_raw = re.findall(r"\[[\s\S]*?\]", text)
    arrays: list[list] = []
    for raw in arrays_raw:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            arrays.append(value)
    return arrays


def parse_input_md(path: Path) -> ParsedInput:
    text = path.read_text(encoding="utf-8")

    base_match = re.search(r"base_prompt:\s*(.*?)\n---", text, re.S)
    if base_match is None:
        raise ValueError("Cannot parse base_prompt from input.md")
    base_prompt = base_match.group(1).strip()

    arrays = _extract_json_arrays(text)
    repo_urls: list[str] | None = None
    scores: list[int] | None = None
    for arr in arrays:
        if arr and all(isinstance(x, str) and x.startswith("http") for x in arr):
            repo_urls = arr
        elif arr and all(str(x) in {"0", "1", "2"} for x in arr):
            scores = [int(x) for x in arr]

    if repo_urls is None:
        raise ValueError("Cannot parse repo url list from input.md")
    if scores is None:
        raise ValueError("Cannot parse score list from input.md")
    if len(repo_urls) != 7 or len(scores) != 7:
        raise ValueError("repo urls and scores must both have length 7")

    remarks: dict[int, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^repo(\d+)\W+(.+)$", stripped)
        if match is None:
            continue
        idx = int(match.group(1))
        note = match.group(2).strip()
        if 1 <= idx <= 7 and note:
            remarks[idx] = note

    return ParsedInput(
        base_prompt=base_prompt,
        repo_urls=repo_urls,
        scores=scores,
        remarks=remarks,
    )


def fetch_repos(data: ParsedInput) -> list[RepoResult]:
    results: list[RepoResult] = []
    for i, url in enumerate(data.repo_urls, start=1):
        repo_id = f"repo{i}"
        score = data.scores[i - 1]
        remark = data.remarks.get(i, "")
        try:
            extracted = extract_page_content(url)
            text = extracted.get("text", "")
            result = RepoResult(
                repo_id=repo_id,
                url=url,
                score=score,
                remark=remark,
                ok=True,
                title=extracted.get("title", ""),
                text_preview=text[:220].replace("\n", " "),
                text_length=int(extracted.get("text_length", 0)),
                error="",
            )
        except Exception as exc:
            result = RepoResult(
                repo_id=repo_id,
                url=url,
                score=score,
                remark=remark,
                ok=False,
                title="",
                text_preview="",
                text_length=0,
                error=str(exc),
            )
        results.append(result)
    return results


def choose_debug_repo(results: list[RepoResult]) -> RepoResult | None:
    with_remark = [
        r for r in results if r.remark and r.score in (0, 1)
    ]
    if with_remark:
        return with_remark[0]

    score_zero = [r for r in results if r.score == 0]
    if score_zero:
        return score_zero[0]

    score_one = [r for r in results if r.score == 1]
    if score_one:
        return score_one[0]
    return None


def choose_new_requirement_repo(results: list[RepoResult]) -> RepoResult | None:
    score_two = [r for r in results if r.score == 2]
    if score_two:
        return score_two[0]
    return None


def build_debug_prompt(repo: RepoResult) -> str:
    note = repo.remark if repo.remark else "存在交互或机制问题"
    return (
        f"请你帮我修复 `{repo.repo_id}` 这个版本。"
        f"当前明显问题是：{note}。"
        "请先定位根因，再完整修复。"
        "修复时不要破坏已可用功能，修完后请你自测完整对局流程，"
        "重点检查落子、boop 推挤、棋子回收和胜负判定。"
    )


def build_new_requirement_prompt(repo: RepoResult) -> str:
    return (
        f"请基于 `{repo.repo_id}` 只新增一个具体功能：`落子结果预览`。"
        "不要改动 Boop 的核心规则（6x6 棋盘、kitten 推挤、三 kitten 升 cat、三 cat 获胜）。"
        "功能细则："
        "1) 当玩家鼠标悬停在可落子格时，实时高亮这一步会被 boop 推动的棋子；"
        "2) 同时高亮这些棋子的目标落点；"
        "3) 如果某棋子会被挤出棋盘，要明确标记“将被挤出”；"
        "4) 玩家点击落子后，实际结果必须与预览一致。"
        "验收标准：任意选一个可落子格，预览信息完整且与最终落子结果一致。"
    )


def build_unable_note(results: list[RepoResult], new_repo: RepoResult | None) -> str:
    if new_repo is not None:
        return "无。已完成新增需求 repo 选择。"

    has_score_two = any(r.score == 2 for r in results)
    if not has_score_two:
        return "没有 2 分 case，无法新增需求。"

    return "2 分 case 无法确认为可用状态，无法新增需求。"


def write_fetch_json(path: Path, data: ParsedInput, results: list[RepoResult]) -> None:
    payload = {
        "base_prompt": data.base_prompt,
        "repo_urls": data.repo_urls,
        "scores": data.scores,
        "remarks": data.remarks,
        "results": [
            {
                "repo": r.repo_id,
                "url": r.url,
                "score": r.score,
                "remark": r.remark,
                "ok": r.ok,
                "title": r.title,
                "text_preview": r.text_preview,
                "text_length": r.text_length,
                "error": r.error,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_output_md(
    path: Path,
    debug_repo: RepoResult | None,
    new_repo: RepoResult | None,
    debug_prompt: str,
    new_prompt: str,
    unable_note: str,
) -> None:
    debug_repo_id = debug_repo.repo_id if debug_repo else ""
    new_repo_id = new_repo.repo_id if new_repo else ""

    content = (
        "# 输出结果\n"
        f"debug prompt:\n{debug_prompt}\n\n"
        f"debug改写repo:\n{debug_repo_id}\n\n"
        f"新增需求prompt:\n{new_prompt}\n\n"
        f"新增需求改写repo:\n{new_repo_id}\n\n"
        f"无法新增需求备注:\n{unable_note}\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_md)
    output_path = Path(args.output_md)
    fetch_path = Path(args.fetch_json)

    parsed = parse_input_md(input_path)
    results = fetch_repos(parsed)
    write_fetch_json(fetch_path, parsed, results)

    debug_repo = choose_debug_repo(results)
    new_repo = choose_new_requirement_repo(results)

    if debug_repo is None:
        debug_prompt = "7 个产物都很优秀，无法写 debug"
    else:
        debug_prompt = build_debug_prompt(debug_repo)

    if new_repo is None:
        new_prompt = ""
    else:
        new_prompt = build_new_requirement_prompt(new_repo)

    unable_note = build_unable_note(results, new_repo)

    write_output_md(
        output_path,
        debug_repo,
        new_repo,
        debug_prompt,
        new_prompt,
        unable_note,
    )
    print(f"Generated: {output_path}")
    print(f"Fetched: {fetch_path}")


if __name__ == "__main__":
    main()
