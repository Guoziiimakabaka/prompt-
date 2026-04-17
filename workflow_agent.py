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
class RuntimeCheck:
    repo_id: str
    checked: bool
    passed: bool
    issues: list[str]


@dataclass
class AnalysisBundle:
    repo_issues: dict[str, list[str]]
    global_issues: list[str]
    runtime_checks: dict[str, RuntimeCheck]


class AgentDecision(BaseModel):
    debug_prompt: str = Field(...)
    debug_rewrite_repo: str = Field(...)
    new_requirement_prompt: str = Field(...)
    new_requirement_rewrite_repo: str = Field(...)
    unable_to_add_requirement_note: str = Field(...)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LLM workflow agent: parse input.md, fetch repo pages, "
            "run runtime checks, and generate output.md."
        )
    )
    parser.add_argument("--input_md", default="input.md", help="Input markdown path.")
    parser.add_argument(
        "--output_md",
        default="output.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--fetch_json",
        default="repo_fetch_results.json",
        help="Fetched repo details path.",
    )
    return parser.parse_args()


def _extract_json_arrays(text: str) -> list[list[Any]]:
    arrays_raw = re.findall(r"\[[\s\S]*?\]", text)
    arrays: list[list[Any]] = []
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

    base_match = re.search(r"base_prompt\s*:\s*(.*?)(?:\n---|\Z)", text, re.S | re.I)
    if base_match is None:
        raise ValueError("Cannot parse base_prompt from input.md")
    base_prompt = base_match.group(1).strip()

    arrays = _extract_json_arrays(text)
    repo_urls: list[str] | None = None
    scores: list[int] | None = None

    for arr in arrays:
        if arr and all(isinstance(x, str) and x.startswith("http") for x in arr):
            repo_urls = arr
            continue

        if arr and all(str(x).strip() in {"0", "1", "2"} for x in arr):
            scores = [int(str(x).strip()) for x in arr]

    if repo_urls is None or scores is None:
        raise ValueError("Cannot parse repo_urls or scores from input.md")
    if len(repo_urls) != 7 or len(scores) != 7:
        raise ValueError("repo_urls and scores must both have length 7")

    remarks: dict[int, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("repo"):
            continue

        match = re.match(r"^repo(\d+)\s*[：:]\s*(.+)$", stripped, re.I)
        if match is None:
            match = re.match(r"^repo(\d+)\s*[-—]\s*(.+)$", stripped, re.I)
        if match is None:
            continue

        index = int(match.group(1))
        note = match.group(2).strip()
        if 1 <= index <= 7 and note:
            remarks[index] = note

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
            results.append(
                RepoResult(
                    repo_id=repo_id,
                    url=url,
                    score=score,
                    remark=remark,
                    ok=True,
                    title=str(extracted.get("title", "")),
                    text_preview=str(text)[:300].replace("\n", " "),
                    text_length=int(extracted.get("text_length", 0)),
                    error="",
                )
            )
        except Exception as exc:
            results.append(
                RepoResult(
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
            )

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


def _repo_by_id(repo_id: str, results: list[RepoResult]) -> RepoResult | None:
    for result in results:
        if result.repo_id == repo_id:
            return result
    return None


def _has_repo_marker(text: str) -> bool:
    return bool(re.search(r"repo\s*\d+", text, re.I))


def _has_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def _has_colloquial(text: str) -> bool:
    colloquial_markers = [
        "帮我",
        "请你",
        "搞一个",
        "弄一个",
        "来个",
        "给我",
        "整一个",
        "顺手",
    ]
    return any(marker in text for marker in colloquial_markers)


def run_runtime_check(repo_id: str, url: str) -> RuntimeCheck:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return RuntimeCheck(
            repo_id=repo_id,
            checked=False,
            passed=False,
            issues=[f"运行检查不可用：未安装 Playwright。{exc}"],
        )

    page_errors: list[str] = []
    console_errors: list[str] = []
    failed_requests: list[str] = []
    issues: list[str] = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(ignore_https_errors=True)
            page = context.new_page()

            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text)
                if msg.type == "error"
                else None,
            )
            page.on("requestfailed", lambda req: failed_requests.append(req.url))

            page.goto(url, wait_until="networkidle", timeout=30000)

            root_rendered = page.evaluate(
                """
                () => {
                  const root = document.querySelector('#root');
                  if (!root) {
                    return false;
                  }
                  const text = (root.textContent || '').trim();
                  return root.childElementCount > 0 || text.length > 0;
                }
                """
            )
            body_text = page.inner_text("body")
            text_len = len((body_text or "").strip())

            if not root_rendered and text_len < 20:
                issues.append("页面运行后接近空白，疑似渲染失败。")
            if page_errors:
                issues.append(f"捕获到页面脚本异常 {len(page_errors)} 条。")
            if console_errors:
                issues.append(f"捕获到控制台错误 {len(console_errors)} 条。")
            if len(failed_requests) >= 3:
                issues.append(f"捕获到网络请求失败 {len(failed_requests)} 条。")

            context.close()
            browser.close()
    except Exception as exc:
        return RuntimeCheck(
            repo_id=repo_id,
            checked=False,
            passed=False,
            issues=[f"运行检查执行失败：{exc}"],
        )

    return RuntimeCheck(
        repo_id=repo_id,
        checked=True,
        passed=not issues,
        issues=issues,
    )


def build_analysis(results: list[RepoResult]) -> AnalysisBundle:
    repo_issues: dict[str, list[str]] = {}
    global_issues: list[str] = []
    runtime_checks: dict[str, RuntimeCheck] = {}

    fetch_failed_repos: list[str] = []
    default_title_repos: list[str] = []
    empty_text_repos: list[str] = []

    for result in results:
        issues: list[str] = []

        if not result.ok:
            fetch_failed_repos.append(result.repo_id)
            issues.append(f"页面访问或解析失败：{result.error or '未知错误'}")
        if result.title.strip().strip('"') in {"", "React App"}:
            default_title_repos.append(result.repo_id)
            issues.append("页面标题疑似默认模板。")
        if result.text_length == 0:
            empty_text_repos.append(result.repo_id)
            issues.append("未提取到正文文本。")

        runtime_result = run_runtime_check(result.repo_id, result.url)
        runtime_checks[result.repo_id] = runtime_result
        if not runtime_result.passed:
            issues.extend(runtime_result.issues)

        if issues:
            repo_issues[result.repo_id] = issues

    if fetch_failed_repos:
        global_issues.append(f"抓取失败页面：{', '.join(fetch_failed_repos)}。")
    if default_title_repos:
        global_issues.append(f"默认标题页面：{', '.join(default_title_repos)}。")
    if empty_text_repos:
        global_issues.append(f"正文为空页面：{', '.join(empty_text_repos)}。")

    return AnalysisBundle(
        repo_issues=repo_issues,
        global_issues=global_issues,
        runtime_checks=runtime_checks,
    )


def _fallback_debug_prompt(
    results: list[RepoResult],
    debug_repo: str,
    analysis: AnalysisBundle,
) -> str:
    selected = _repo_by_id(debug_repo, results)
    remark = selected.remark.strip() if selected and selected.remark.strip() else ""

    issue_list = analysis.repo_issues.get(debug_repo, [])
    issue_text = "；".join(issue_list[:3]) if issue_list else ""

    problem_desc = remark or issue_text or "当前实现存在规则执行与界面状态不一致问题"
    if issue_text and remark and issue_text not in problem_desc:
        problem_desc = f"{problem_desc}；并发现：{issue_text}"

    return (
        f"当前版本存在以下问题：{problem_desc}。"
        "请完成根因定位与修复，确保规则判定、状态流转与交互反馈一致。"
        "请提供最小复现步骤、修复方案、关键代码改动点，以及覆盖正常流程和边界条件的回归验证结果。"
    )


def _fallback_new_requirement_prompt() -> str:
    return (
        "在不改变现有核心规则的前提下，新增“落子结果预览”功能。"
        "功能要求：当鼠标悬停在可落子位置时，实时高亮即将被推移的棋子与目标落点；"
        "若目标落点越界，应显示明确的越界提示并标注被移出棋盘的棋子。"
        "验收标准：随机选择三个可落子位置进行验证，预览结果与实际执行结果逐项一致；"
        "开启与关闭预览功能后，现有回合流程、胜负判定与操作响应保持一致。"
    )


def rewrite_prompt_to_formal_chinese(
    llm: ChatOpenAI,
    prompt_text: str,
    purpose: str,
) -> str:
    system_prompt = (
        "你是提示词改写助手。"
        "请将输入改写为正式、具体、可执行的中文提示词。"
        "不得出现英文，不得出现 repo 编号，不得出现网址，不得口语化。"
        "只返回改写后的文本。"
    )
    human_prompt = f"用途：{purpose}\n原文：{prompt_text}"

    result = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    return str(result.content).strip()


def _sort_repos(repos: list[RepoResult]) -> list[RepoResult]:
    return sorted(repos, key=lambda r: (r.score, r.repo_id))


def validate_and_fix_decision(
    decision: AgentDecision,
    results: list[RepoResult],
    llm: ChatOpenAI,
    analysis: AnalysisBundle,
) -> AgentDecision:
    debug_repo = decision.debug_rewrite_repo.strip()
    new_repo = decision.new_requirement_rewrite_repo.strip()

    remark_candidates = _sort_repos([r for r in results if r.remark.strip()])
    score0 = _sort_repos([r for r in results if r.score == 0])
    score1 = _sort_repos([r for r in results if r.score == 1])
    score2 = _sort_repos([r for r in results if r.score == 2])

    selected_debug = _repo_by_id(debug_repo, results) if debug_repo else None
    if selected_debug is None:
        if remark_candidates:
            debug_repo = remark_candidates[0].repo_id
        elif score0:
            debug_repo = score0[0].repo_id
        elif score1:
            debug_repo = score1[0].repo_id
        else:
            debug_repo = ""
    else:
        has_remark = bool(selected_debug.remark.strip())
        if remark_candidates and not has_remark:
            debug_repo = remark_candidates[0].repo_id
        elif (
            selected_debug.score not in {0, 1}
            and not has_remark
            and (score0 or score1)
        ):
            debug_repo = (score0 + score1)[0].repo_id

    eligible_score2: list[str] = []
    for repo in score2:
        runtime_result = analysis.runtime_checks.get(repo.repo_id)
        if runtime_result and runtime_result.checked and runtime_result.passed:
            eligible_score2.append(repo.repo_id)

    selected_new = _repo_by_id(new_repo, results) if new_repo else None
    if (
        selected_new is None
        or selected_new.score != 2
        or selected_new.repo_id not in eligible_score2
    ):
        new_repo = eligible_score2[0] if eligible_score2 else ""

    debug_prompt = decision.debug_prompt.strip()
    if not debug_prompt:
        if debug_repo:
            debug_prompt = _fallback_debug_prompt(results, debug_repo, analysis)
        else:
            debug_prompt = "7 个产物都很优秀，无法写 debug"

    new_prompt = decision.new_requirement_prompt.strip()
    unable_note = decision.unable_to_add_requirement_note.strip()

    if new_repo:
        if not new_prompt:
            new_prompt = _fallback_new_requirement_prompt()
        if not unable_note:
            unable_note = "无。已选择运行检查通过的 2 分产物用于新增需求。"
    else:
        new_prompt = ""
        if not unable_note:
            if score2:
                unable_note = (
                    "所有 2 分 repo 经运行检查后均存在明显 bug "
                    "或无法完成运行验证，无法新增需求。"
                )
            else:
                unable_note = "没有可用的 2 分 repo，无法新增需求。"

    if (
        _has_repo_marker(debug_prompt)
        or _has_english(debug_prompt)
        or _has_colloquial(debug_prompt)
    ):
        debug_prompt = rewrite_prompt_to_formal_chinese(llm, debug_prompt, "调试提示词")
    if new_prompt and (
        _has_repo_marker(new_prompt)
        or _has_english(new_prompt)
        or _has_colloquial(new_prompt)
    ):
        new_prompt = rewrite_prompt_to_formal_chinese(llm, new_prompt, "新增需求提示词")

    if (
        _has_repo_marker(debug_prompt)
        or _has_english(debug_prompt)
        or _has_colloquial(debug_prompt)
    ):
        if debug_repo:
            debug_prompt = _fallback_debug_prompt(results, debug_repo, analysis)
        else:
            debug_prompt = "7 个产物都很优秀，无法写 debug"

    if new_prompt and (
        _has_repo_marker(new_prompt)
        or _has_english(new_prompt)
        or _has_colloquial(new_prompt)
    ):
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
    analysis: AnalysisBundle,
) -> str:
    cases: list[dict[str, Any]] = []

    for result in results:
        runtime_result = analysis.runtime_checks.get(result.repo_id)
        cases.append(
            {
                "repo_id": result.repo_id,
                "url": result.url,
                "score": result.score,
                "remark": result.remark,
                "fetch_ok": result.ok,
                "title": result.title,
                "text_preview": result.text_preview,
                "text_length": result.text_length,
                "error": result.error,
                "extra_issues": analysis.repo_issues.get(result.repo_id, []),
                "runtime_checked": runtime_result.checked if runtime_result else False,
                "runtime_passed": runtime_result.passed if runtime_result else False,
                "runtime_issues": runtime_result.issues if runtime_result else [],
            }
        )

    payload = {
        "base_prompt": parsed.base_prompt,
        "rules": {
            "debug_select_priority": (
                "优先选择有具体 remark 的 repo；若无，再依次选择 0 分、1 分；"
                "debug 只选一个 repo。"
            ),
            "new_requirement_select": (
                "新增需求改写 repo 只能选择 2 分且运行检查通过的 repo；若没有则留空。"
            ),
            "prompt_language": "debug_prompt 和 new_requirement_prompt 必须是中文。",
            "prompt_content": "两个 prompt 文本禁止出现 repo 编号、网址和英文。",
            "style": "两个 prompt 必须正式、具体、非口语化。",
            "new_requirement_prompt": "新增需求必须是单一具体功能，并包含验收标准。",
        },
        "global_extra_issues": analysis.global_issues,
        "cases": cases,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm(
    parsed: ParsedInput,
    results: list[RepoResult],
    analysis: AnalysisBundle,
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
        "你是用于生成质检提示词的 Agent。"
        "你必须输出 JSON 对象。"
        "debug 和新增需求各只选一个 repo。"
        "新增需求改写 repo 只能从“2 分且运行检查通过”的 repo 中选择；若无可选项必须留空。"
        "两个 prompt 必须为中文，且正式、具体、非口语化，不得出现 repo 编号和网址。"
        "新增需求 prompt 必须是单一具体功能，并包含明确验收标准。"
        "请结合 extra_issues、runtime_issues、global_extra_issues 进行判断。"
    )
    human_prompt = (
        "请根据以下输入生成 JSON 对象。"
        "只返回 JSON，不要解释，不要代码块。\n"
        f"{build_llm_input(parsed, results, analysis)}"
    )

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    content = str(response.content).strip()

    fenced_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
    if fenced_match:
        content = fenced_match.group(1)

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
    fetch_json_path = Path(args.fetch_json)

    parsed = parse_input_md(input_path)
    results = fetch_repos(parsed)
    write_fetch_json(fetch_json_path, parsed, results)
    analysis = build_analysis(results)
    decision = call_llm(parsed, results, analysis)
    write_output_md(output_path, decision)

    print(f"Generated: {output_path}")
    print(f"Fetched: {fetch_json_path}")


if __name__ == "__main__":
    main()
