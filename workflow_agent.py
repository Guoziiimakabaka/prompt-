import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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


@dataclass
class FetchAnalysis:
    repo_issues: dict[str, list[str]]
    global_issues: list[str]


class AgentDecision(BaseModel):
    debug_prompt: str = Field(...)
    debug_rewrite_repo: str = Field(...)
    new_requirement_prompt: str = Field(...)
    new_requirement_rewrite_repo: str = Field(...)
    unable_to_add_requirement_note: str = Field(...)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM workflow agent: parse input.md, fetch repo pages, generate output.md."
    )
    parser.add_argument("--input_md", default="input.md", help="Input markdown path.")
    parser.add_argument("--output_md", default="output.md", help="Output markdown path.")
    parser.add_argument(
        "--fetch_json",
        default="repo_fetch_results.json",
        help="Fetched repo details path.",
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
        if not stripped.lower().startswith("repo"):
            continue
        m = re.match(r"^repo(\d+)\s*[：:]\s*(.+)$", stripped)
        if m is None:
            m = re.match(r"^repo(\d+)\W(.+)$", stripped)
        if m is None:
            continue
        idx = int(m.group(1))
        note = m.group(2).strip()
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
                text_preview=text[:300].replace("\n", " "),
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


def write_fetch_json(path: Path, data: ParsedInput, results: list[RepoResult]) -> None:
    payload = {
        "base_prompt": data.base_prompt,
        "repo_urls": data.repo_urls,
        "scores": data.scores,
        "remarks": data.remarks,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_fetch_json(path: Path) -> FetchAnalysis:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("results", [])
    repo_issues: dict[str, list[str]] = {}
    global_issues: list[str] = []

    failed: list[str] = []
    default_title: list[str] = []
    empty_text: list[str] = []
    likely_csr: list[str] = []

    for item in items:
        repo_id = str(item.get("repo_id", "")).strip()
        if not repo_id:
            continue
        issues: list[str] = []
        ok = bool(item.get("ok", False))
        title = str(item.get("title", "")).strip().strip('"')
        text_length = int(item.get("text_length", 0) or 0)
        error = str(item.get("error", "")).strip()

        if not ok:
            failed.append(repo_id)
            issues.append(f"页面访问或解析失败：{error or '未知错误'}")
        if title in {"React App", ""}:
            default_title.append(repo_id)
            issues.append("页面标题未定制，疑似默认模板。")
        if text_length == 0:
            empty_text.append(repo_id)
            issues.append("未提取到正文文本。")
            if ok and title not in {"", "React App"}:
                likely_csr.append(repo_id)
                issues.append("疑似依赖脚本渲染，首屏静态内容不足。")

        if issues:
            repo_issues[repo_id] = issues

    if failed:
        global_issues.append(f"抓取失败页面：{', '.join(failed)}。")
    if default_title:
        global_issues.append(f"默认标题页面：{', '.join(default_title)}。")
    if empty_text:
        global_issues.append(f"正文为空页面：{', '.join(empty_text)}。")
    if likely_csr:
        global_issues.append(f"疑似前端脚本渲染主导页面：{', '.join(likely_csr)}。")

    return FetchAnalysis(repo_issues=repo_issues, global_issues=global_issues)


def _repo_by_id(repo_id: str, results: list[RepoResult]) -> RepoResult | None:
    for r in results:
        if r.repo_id == repo_id:
            return r
    return None


def _has_repo_marker(text: str) -> bool:
    return bool(re.search(r"repo\s*\d+", text, re.IGNORECASE))


def _has_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def _has_colloquial(text: str) -> bool:
    markers = ["吧", "一下", "帮我", "请你", "搞一下", "弄一下"]
    return any(m in text for m in markers)


def _clean_note(note: str) -> str:
    cleaned = note.strip()
    if not cleaned:
        return "游戏核心机制存在实现错误"
    return cleaned


def _fallback_debug_prompt(
    results: list[RepoResult],
    debug_repo: str,
    analysis: FetchAnalysis,
) -> str:
    selected = _repo_by_id(debug_repo, results)
    note = _clean_note(selected.remark if selected else "")
    extras = "；".join(analysis.repo_issues.get(debug_repo, [])[:2])
    extra_clause = f"；需同步核查：{extras}" if extras else ""
    return (
        f"当前版本存在以下问题：{note}{extra_clause}。"
        "请完成根因定位与修复，并提供回归验证结果。"
        "验证范围至少包括：落子行为、规则判定、棋子状态流转、异常边界场景。"
    )


def _fallback_new_requirement_prompt() -> str:
    return (
        "在不改变现有核心规则的前提下，新增“落子结果预览”功能。"
        "功能要求：当鼠标悬停于可落子位置时，实时展示受影响棋子与目标落点；"
        "若目标落点越界，需明确提示将被移出棋盘。"
        "验收标准：随机选择三个可落子位置，预览结果与实际执行结果逐项一致。"
    )


def rewrite_prompt_to_formal_chinese(
    llm: ChatOpenAI,
    prompt_text: str,
    purpose: str,
) -> str:
    system_prompt = (
        "你是提示词改写助手。"
        "请将输入改写为正式、具体、可执行的中文提示词。"
        "必须满足：不得出现英文、不得出现repo编号、不得出现网址、不得口语化。"
        "仅返回改写后的文本。"
    )
    human_prompt = f"用途：{purpose}\n原文：{prompt_text}"
    result = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    return str(result.content).strip()


def validate_and_fix_decision(
    decision: AgentDecision,
    results: list[RepoResult],
    llm: ChatOpenAI,
    analysis: FetchAnalysis,
) -> AgentDecision:
    debug_repo = decision.debug_rewrite_repo.strip()
    new_repo = decision.new_requirement_rewrite_repo.strip()

    remark_candidates = [r for r in results if r.remark]
    score01 = [r for r in results if r.score in (0, 1)]
    score2 = [r for r in results if r.score == 2]

    # Debug repo rule.
    selected_debug = _repo_by_id(debug_repo, results) if debug_repo else None
    if selected_debug is None:
        if remark_candidates:
            debug_repo = remark_candidates[0].repo_id
        elif score01:
            debug_repo = score01[0].repo_id
        else:
            debug_repo = ""
    else:
        if selected_debug.remark == "" and remark_candidates:
            debug_repo = remark_candidates[0].repo_id
        elif selected_debug.score not in (0, 1):
            debug_repo = score01[0].repo_id if score01 else ""

    # New requirement repo rule: must be score=2.
    selected_new = _repo_by_id(new_repo, results) if new_repo else None
    if selected_new is None or selected_new.score != 2:
        new_repo = score2[0].repo_id if score2 else ""

    debug_prompt = decision.debug_prompt.strip()
    if not debug_prompt:
        debug_prompt = (
            _fallback_debug_prompt(results, debug_repo, analysis)
            if debug_repo
            else "7 个产物都很优秀，无法写 debug"
        )

    new_prompt = decision.new_requirement_prompt.strip()
    unable_note = decision.unable_to_add_requirement_note.strip()
    if new_repo:
        if not new_prompt:
            new_prompt = _fallback_new_requirement_prompt()
        if not unable_note:
            unable_note = "无。已完成新增需求 repo 选择。"
    else:
        new_prompt = ""
        if not unable_note:
            unable_note = "7个repo都有逻辑错误，无法新增需求"

    # Enforce style.
    if _has_repo_marker(debug_prompt) or _has_english(debug_prompt) or _has_colloquial(debug_prompt):
        rewritten = rewrite_prompt_to_formal_chinese(llm, debug_prompt, "调试提示词")
        debug_prompt = rewritten if rewritten else _fallback_debug_prompt(results, debug_repo, analysis)

    if _has_repo_marker(new_prompt) or _has_english(new_prompt) or _has_colloquial(new_prompt):
        rewritten = rewrite_prompt_to_formal_chinese(llm, new_prompt, "新增需求提示词")
        new_prompt = rewritten if rewritten else _fallback_new_requirement_prompt()

    # Final fallback.
    if _has_repo_marker(debug_prompt) or _has_english(debug_prompt) or _has_colloquial(debug_prompt):
        debug_prompt = (
            _fallback_debug_prompt(results, debug_repo, analysis)
            if debug_repo
            else "7 个产物都很优秀，无法写 debug"
        )
    if new_prompt and (_has_repo_marker(new_prompt) or _has_english(new_prompt) or _has_colloquial(new_prompt)):
        new_prompt = _fallback_new_requirement_prompt()

    return AgentDecision(
        debug_prompt=debug_prompt,
        debug_rewrite_repo=debug_repo,
        new_requirement_prompt=new_prompt,
        new_requirement_rewrite_repo=new_repo,
        unable_to_add_requirement_note=unable_note,
    )


def build_llm_input(
    parsed: ParsedInput,
    results: list[RepoResult],
    analysis: FetchAnalysis,
) -> str:
    cases: list[dict[str, Any]] = []
    for r in results:
        cases.append(
            {
                "repo_id": r.repo_id,
                "url": r.url,
                "score": r.score,
                "remark": r.remark,
                "fetch_ok": r.ok,
                "title": r.title,
                "text_preview": r.text_preview,
                "text_length": r.text_length,
                "error": r.error,
                "extra_issues": analysis.repo_issues.get(r.repo_id, []),
            }
        )

    payload = {
        "base_prompt": parsed.base_prompt,
        "rules": {
            "debug_select_priority": "优先选择有具体remark的repo；若无remark，再选0分；再选1分；debug只选1个repo",
            "new_requirement_select": "新增需求改写repo只能是2分repo；若无2分repo必须留空",
            "prompt_language": "debug_prompt和new_requirement_prompt必须是中文",
            "prompt_content": "两个prompt文本中禁止出现repo编号",
            "style": "两个prompt必须正式、具体、非口语化",
            "new_requirement_prompt": "新增需求必须是单一具体功能，且包含验收标准",
        },
        "global_extra_issues": analysis.global_issues,
        "cases": cases,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm(
    parsed: ParsedInput,
    results: list[RepoResult],
    analysis: FetchAnalysis,
) -> AgentDecision:
    load_dotenv()
    base_url = os.getenv("MODEL_BASE_URL")
    api_key = os.getenv("MODEL_API_KEY")
    model_name = os.getenv("MODEL_NAME")
    if not base_url or not api_key or not model_name:
        raise ValueError("MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME must be set in .env")

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        api_key=api_key,
        base_url=base_url,
    )

    system_prompt = (
        "你是用于生成质检提示词的Agent。"
        "你必须输出JSON对象。"
        "debug和新增需求各只选一个repo。"
        "新增需求改写repo只能是2分repo；若无2分repo则留空。"
        "两个prompt必须中文，不能出现repo编号，且风格正式、具体、非口语化。"
        "新增需求prompt必须是单一具体功能，并给出验收标准。"
        "请结合extra_issues与global_extra_issues分析是否存在额外问题。"
    )
    human_prompt = (
        "请根据以下输入生成JSON对象。"
        "只返回JSON，不要解释，不要代码块。\n"
        f"{build_llm_input(parsed, results, analysis)}"
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    content = str(response.content).strip()

    fence_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
    if fence_match:
        content = fence_match.group(1)

    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError(f"LLM output must be a JSON object: {raw}")

    normalized = {
        "debug_prompt": raw.get("debug_prompt", ""),
        "debug_rewrite_repo": raw.get("debug_rewrite_repo", raw.get("debug_repo_id", "")),
        "new_requirement_prompt": raw.get(
            "new_requirement_prompt",
            raw.get("new_requirement", {}).get("prompt", "")
            if isinstance(raw.get("new_requirement"), dict)
            else "",
        ),
        "new_requirement_rewrite_repo": raw.get(
            "new_requirement_rewrite_repo",
            raw.get(
                "new_requirement_repo_id",
                raw.get("new_requirement", {}).get("repo_id", "")
                if isinstance(raw.get("new_requirement"), dict)
                else "",
            ),
        ),
        "unable_to_add_requirement_note": raw.get(
            "unable_to_add_requirement_note",
            raw.get("unable_note", ""),
        ),
    }

    decision = AgentDecision.model_validate(normalized)
    return validate_and_fix_decision(decision, results, llm, analysis)


def write_output_md(path: Path, decision: AgentDecision) -> None:
    content = (
        "# 输出结果\n"
        f"debug prompt:\n{decision.debug_prompt}\n\n"
        f"debug改写repo:\n{decision.debug_rewrite_repo}\n\n"
        f"新增需求prompt:\n{decision.new_requirement_prompt}\n\n"
        f"新增需求改写repo:\n{decision.new_requirement_rewrite_repo}\n\n"
        f"无法新增需求备注:\n{decision.unable_to_add_requirement_note}\n"
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
    analysis = analyze_fetch_json(fetch_path)
    decision = call_llm(parsed, results, analysis)
    write_output_md(output_path, decision)

    print(f"Generated: {output_path}")
    print(f"Fetched: {fetch_path}")


if __name__ == "__main__":
    main()
