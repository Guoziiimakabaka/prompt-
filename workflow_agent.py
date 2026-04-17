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
from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import BaseModel, Field

from app import extract_page_content


INPUT_HEADERS = {
    "uid",
    "prompt",
    "repo",
    "评分",
    "产物备注（参考）",
}

OUTPUT_HEADERS = [
    "debug prompt",
    "debug 改写repo",
    "无法写debug改写备注",
    "新增需求 prompt",
    "新增需求改写 repo",
    "无法新增需求备注",
]

DEBUG_UNABLE_TEXT = "7 个产物都很优秀，无法写 debug"


@dataclass
class ParsedInput:
    uid: str
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
            "Read cases from input.xlsx, fetch 7 repos per case, "
            "run runtime checks, generate prompts, and write back into Excel."
        )
    )
    parser.add_argument("--input_xlsx", default="input.xlsx", help="Input Excel path.")
    parser.add_argument(
        "--output_xlsx",
        default="",
        help="Output Excel path. Empty means overwrite input file.",
    )
    parser.add_argument(
        "--sheet_name",
        default="",
        help="Sheet name to process. Empty means the first sheet.",
    )
    parser.add_argument(
        "--start_row",
        type=int,
        default=2,
        help="Start row index (1-based).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Max data rows to process. 0 means no limit.",
    )
    parser.add_argument(
        "--uid",
        default="",
        help="Only process the row whose uid matches this value.",
    )
    parser.add_argument(
        "--fetch_json",
        default="repo_fetch_results.json",
        help="Write fetched/runtime details to this JSON file.",
    )
    return parser.parse_args()


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set in .env")
    return value


def _header_map(ws: Worksheet) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for col in range(1, ws.max_column + 1):
        header = _to_text(ws.cell(row=1, column=col).value)
        if header:
            mapping[header] = col
    return mapping


def _ensure_output_headers(ws: Worksheet, mapping: dict[str, int]) -> dict[str, int]:
    next_col = ws.max_column + 1
    for header in OUTPUT_HEADERS:
        if header not in mapping:
            ws.cell(row=1, column=next_col).value = header
            mapping[header] = next_col
            next_col += 1
    return mapping


def _assert_input_headers(mapping: dict[str, int]) -> None:
    missing = [header for header in INPUT_HEADERS if header not in mapping]
    if missing:
        raise ValueError(f"Missing required headers in Excel: {missing}")


def _parse_json_list(raw: str, field_name: str) -> list[Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON list: {raw}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON list")
    return value


def _parse_urls(raw: str) -> list[str]:
    values = _parse_json_list(raw, "repo")
    if len(values) != 7:
        raise ValueError("repo URL list must contain exactly 7 URLs")

    urls: list[str] = []
    for v in values:
        text = _to_text(v)
        if not text.startswith("http"):
            raise ValueError(f"Invalid repo URL: {text}")
        urls.append(text)
    return urls


def _parse_scores(raw: str) -> list[int]:
    values = _parse_json_list(raw, "评分")
    if len(values) != 7:
        raise ValueError("评分 list must contain exactly 7 items")

    scores: list[int] = []
    for v in values:
        value = int(_to_text(v))
        if value not in {0, 1, 2}:
            raise ValueError(f"评分 item must be 0/1/2, got: {value}")
        scores.append(value)
    return scores


def _parse_remarks(raw: str) -> dict[int, str]:
    if not raw:
        return {}

    normalized = raw.replace("；", "\n").replace(";", "\n")
    remarks: dict[int, str] = {}
    for line in normalized.splitlines():
        text = line.strip()
        if not text:
            continue
        match = re.match(r"repo\s*([1-7])\s*[：:]\s*(.*)$", text, re.I)
        if not match:
            continue
        idx = int(match.group(1))
        note = match.group(2).strip()
        if note:
            remarks[idx] = note
    return remarks


def parse_row_input(ws: Worksheet, row_idx: int, mapping: dict[str, int]) -> ParsedInput | None:
    uid = _to_text(ws.cell(row=row_idx, column=mapping["uid"]).value)
    prompt = _to_text(ws.cell(row=row_idx, column=mapping["prompt"]).value)
    repo_raw = _to_text(ws.cell(row=row_idx, column=mapping["repo"]).value)
    score_raw = _to_text(ws.cell(row=row_idx, column=mapping["评分"]).value)
    remark_raw = _to_text(ws.cell(row=row_idx, column=mapping["产物备注（参考）"]).value)

    if not uid and not prompt and not repo_raw and not score_raw:
        return None

    if not uid:
        raise ValueError(f"Row {row_idx}: uid is empty")
    if not prompt:
        raise ValueError(f"Row {row_idx}: prompt is empty")
    if not repo_raw:
        raise ValueError(f"Row {row_idx}: repo is empty")
    if not score_raw:
        raise ValueError(f"Row {row_idx}: 评分 is empty")

    return ParsedInput(
        uid=uid,
        base_prompt=prompt,
        repo_urls=_parse_urls(repo_raw),
        scores=_parse_scores(score_raw),
        remarks=_parse_remarks(remark_raw),
    )


def _repo_by_id(repo_id: str, results: list[RepoResult]) -> RepoResult | None:
    for result in results:
        if result.repo_id == repo_id:
            return result
    return None


def fetch_repos(data: ParsedInput) -> list[RepoResult]:
    results: list[RepoResult] = []
    for i, url in enumerate(data.repo_urls, start=1):
        repo_id = f"repo{i}"
        score = data.scores[i - 1]
        remark = data.remarks.get(i, "")
        try:
            extracted = extract_page_content(url)
            text = _to_text(extracted.get("text", ""))
            results.append(
                RepoResult(
                    repo_id=repo_id,
                    url=url,
                    score=score,
                    remark=remark,
                    ok=True,
                    title=_to_text(extracted.get("title", "")),
                    text_preview=text[:300].replace("\n", " "),
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
                  if (!root) return false;
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

    failed_fetch: list[str] = []
    default_title: list[str] = []
    empty_text: list[str] = []

    for result in results:
        issues: list[str] = []
        if not result.ok:
            failed_fetch.append(result.repo_id)
            issues.append(f"页面访问或解析失败：{result.error or '未知错误'}")
        if result.title.strip().strip('"') in {"", "React App"}:
            default_title.append(result.repo_id)
            issues.append("页面标题疑似默认模板。")
        if result.text_length == 0:
            empty_text.append(result.repo_id)
            issues.append("未提取到正文文本。")

        runtime = run_runtime_check(result.repo_id, result.url)
        runtime_checks[result.repo_id] = runtime
        if not runtime.passed:
            issues.extend(runtime.issues)

        if issues:
            repo_issues[result.repo_id] = issues

    if failed_fetch:
        global_issues.append(f"抓取失败页面：{', '.join(failed_fetch)}。")
    if default_title:
        global_issues.append(f"默认标题页面：{', '.join(default_title)}。")
    if empty_text:
        global_issues.append(f"正文为空页面：{', '.join(empty_text)}。")

    return AnalysisBundle(
        repo_issues=repo_issues,
        global_issues=global_issues,
        runtime_checks=runtime_checks,
    )


def _has_repo_marker(text: str) -> bool:
    return bool(re.search(r"repo\s*\d+", text, re.I))


def _has_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def _has_colloquial(text: str) -> bool:
    markers = ["帮我", "请你", "搞一个", "弄一个", "来个", "给我", "整一个"]
    return any(m in text for m in markers)


def rewrite_prompt_to_formal_chinese(
    llm: ChatOpenAI,
    prompt_text: str,
    purpose: str,
) -> str:
    system_prompt = (
        "你是提示词改写助手。"
        "请将输入改写为正式、具体、可执行的中文提示词。"
        "不得出现英文、不得出现repo编号、不得出现网址、不得口语化。"
        "只返回改写后的文本。"
    )
    human_prompt = f"用途：{purpose}\n原文：{prompt_text}"
    result = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    return _to_text(result.content)


def _sort_repos(items: list[RepoResult]) -> list[RepoResult]:
    return sorted(items, key=lambda x: (x.score, x.repo_id))


def _has_meaningful_remark(text: str) -> bool:
    cleaned = text.strip()
    return cleaned not in {"", "无", "none", "None", "N/A", "na"}


def _fallback_debug_prompt(
    results: list[RepoResult],
    debug_repo: str,
    analysis: AnalysisBundle,
) -> str:
    selected = _repo_by_id(debug_repo, results)
    note = selected.remark.strip() if selected else ""
    extra = "；".join(analysis.repo_issues.get(debug_repo, [])[:3])
    problem_desc = note if note else "当前版本存在规则判定与交互反馈不一致问题"
    if extra:
        problem_desc = f"{problem_desc}；并发现：{extra}"

    return (
        f"当前版本存在以下问题：{problem_desc}。"
        "请完成根因定位与修复，确保规则判定、状态流转和交互反馈一致。"
        "请提供最小复现步骤、修复方案、关键改动点及覆盖边界场景的回归验证结果。"
    )


def _fallback_new_requirement_prompt() -> str:
    return (
        "在不改变现有核心规则的前提下，新增“落子结果预览”功能。"
        "功能要求：当鼠标悬停在可落子位置时，实时高亮即将被推移的棋子与目标落点；"
        "若目标落点越界，应显示越界提示并标注被移出棋盘的棋子。"
        "验收标准：随机选择三个可落子位置进行验证，预览结果与实际执行结果逐项一致；"
        "开启与关闭预览功能后，现有回合流程、胜负判定与交互响应保持一致。"
    )


def validate_and_fix_decision(
    decision: AgentDecision,
    results: list[RepoResult],
    llm: ChatOpenAI,
    analysis: AnalysisBundle,
) -> AgentDecision:
    debug_repo = decision.debug_rewrite_repo.strip()
    new_repo = decision.new_requirement_rewrite_repo.strip()

    remark_candidates = _sort_repos(
        [r for r in results if _has_meaningful_remark(r.remark)]
    )
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
        selected_has_remark = _has_meaningful_remark(selected_debug.remark)
        if remark_candidates and not selected_has_remark:
            debug_repo = remark_candidates[0].repo_id
        elif (
            selected_debug.score not in {0, 1}
            and not selected_has_remark
            and (score0 or score1)
        ):
            debug_repo = (score0 + score1)[0].repo_id

    eligible_score2: list[str] = []
    for item in score2:
        runtime = analysis.runtime_checks.get(item.repo_id)
        if runtime and runtime.checked and runtime.passed:
            eligible_score2.append(item.repo_id)

    selected_new = _repo_by_id(new_repo, results) if new_repo else None
    if (
        selected_new is None
        or selected_new.score != 2
        or selected_new.repo_id not in eligible_score2
    ):
        new_repo = eligible_score2[0] if eligible_score2 else ""

    debug_prompt = decision.debug_prompt.strip()
    if not debug_prompt:
        debug_prompt = (
            _fallback_debug_prompt(results, debug_repo, analysis)
            if debug_repo
            else DEBUG_UNABLE_TEXT
        )

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

    if _has_repo_marker(debug_prompt) or _has_english(debug_prompt) or _has_colloquial(debug_prompt):
        debug_prompt = rewrite_prompt_to_formal_chinese(llm, debug_prompt, "调试提示词")
    if new_prompt and (_has_repo_marker(new_prompt) or _has_english(new_prompt) or _has_colloquial(new_prompt)):
        new_prompt = rewrite_prompt_to_formal_chinese(llm, new_prompt, "新增需求提示词")

    if _has_repo_marker(debug_prompt) or _has_english(debug_prompt) or _has_colloquial(debug_prompt):
        debug_prompt = (
            _fallback_debug_prompt(results, debug_repo, analysis)
            if debug_repo
            else DEBUG_UNABLE_TEXT
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


def build_llm_input(parsed: ParsedInput, results: list[RepoResult], analysis: AnalysisBundle) -> str:
    cases: list[dict[str, Any]] = []
    for result in results:
        runtime = analysis.runtime_checks.get(result.repo_id)
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
                "runtime_checked": runtime.checked if runtime else False,
                "runtime_passed": runtime.passed if runtime else False,
                "runtime_issues": runtime.issues if runtime else [],
            }
        )

    payload = {
        "uid": parsed.uid,
        "base_prompt": parsed.base_prompt,
        "rules": {
            "debug_select_priority": "优先选择有具体remark的repo；若无remark，再依次选择0分和1分；只选1个。",
            "new_requirement_select": "新增需求改写repo只能选择2分且运行检查通过的repo；若没有则留空。",
            "prompt_language": "debug_prompt和new_requirement_prompt必须是中文。",
            "prompt_content": "两个prompt禁止出现repo编号、网址和英文。",
            "style": "两个prompt必须正式、具体、非口语化。",
            "new_requirement_prompt": "新增需求必须是单一具体功能，并包含明确验收标准。",
        },
        "global_extra_issues": analysis.global_issues,
        "cases": cases,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_llm_json(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        text = fenced.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output must be a JSON object")
    return parsed


def decide_for_case(
    llm: ChatOpenAI,
    parsed: ParsedInput,
    results: list[RepoResult],
    analysis: AnalysisBundle,
) -> AgentDecision:
    system_prompt = (
        "你是用于生成质检提示词的Agent。"
        "你必须输出JSON对象。"
        "debug和新增需求各只选一个repo。"
        "新增需求改写repo只能从“2分且运行检查通过”的repo中选择，若无可选项必须留空。"
        "两个prompt必须是中文、正式、具体、非口语化，且不能出现repo编号、网址和英文。"
        "新增需求prompt必须是一个具体功能，并给出验收标准。"
        "请结合extra_issues、runtime_issues、global_extra_issues综合判断。"
    )
    human_prompt = (
        "请根据以下输入返回JSON对象。"
        "只返回JSON，不要解释，不要代码块。\n"
        f"{build_llm_input(parsed, results, analysis)}"
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    raw = parse_llm_json(_to_text(response.content))

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


def _debug_unable_note(decision: AgentDecision) -> str:
    if not decision.debug_rewrite_repo.strip():
        return DEBUG_UNABLE_TEXT
    if decision.debug_prompt.strip() == DEBUG_UNABLE_TEXT:
        return DEBUG_UNABLE_TEXT
    return "无"


def _build_chat_model() -> ChatOpenAI:
    load_dotenv()
    base_url = _require_env("MODEL_BASE_URL")
    model_name = _require_env("MODEL_NAME")
    api_key = _require_env("MODEL_API_KEY")
    return ChatOpenAI(
        model=model_name,
        temperature=0.2,
        api_key=api_key,
        base_url=base_url,
    )


def process_excel(args: argparse.Namespace) -> None:
    input_path = Path(args.input_xlsx)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_path}")

    output_path = Path(args.output_xlsx) if args.output_xlsx else input_path
    wb = load_workbook(input_path)
    ws = wb[args.sheet_name] if args.sheet_name else wb[wb.sheetnames[0]]

    mapping = _header_map(ws)
    _assert_input_headers(mapping)
    mapping = _ensure_output_headers(ws, mapping)

    llm = _build_chat_model()
    processed = 0
    all_fetch_payloads: list[dict[str, Any]] = []

    for row_idx in range(args.start_row, ws.max_row + 1):
        if args.max_rows > 0 and processed >= args.max_rows:
            break

        row_input = parse_row_input(ws, row_idx, mapping)
        if row_input is None:
            continue
        if args.uid and row_input.uid != args.uid:
            continue

        print(f"[Row {row_idx}] uid={row_input.uid}: fetching repos...")
        repo_results = fetch_repos(row_input)
        analysis = build_analysis(repo_results)

        print(f"[Row {row_idx}] uid={row_input.uid}: calling LLM...")
        decision = decide_for_case(llm, row_input, repo_results, analysis)

        ws.cell(row=row_idx, column=mapping["debug prompt"]).value = decision.debug_prompt
        ws.cell(row=row_idx, column=mapping["debug 改写repo"]).value = decision.debug_rewrite_repo
        ws.cell(row=row_idx, column=mapping["无法写debug改写备注"]).value = _debug_unable_note(decision)
        ws.cell(row=row_idx, column=mapping["新增需求 prompt"]).value = decision.new_requirement_prompt
        ws.cell(row=row_idx, column=mapping["新增需求改写 repo"]).value = decision.new_requirement_rewrite_repo
        ws.cell(row=row_idx, column=mapping["无法新增需求备注"]).value = (
            decision.unable_to_add_requirement_note
        )

        all_fetch_payloads.append(
            {
                "row": row_idx,
                "uid": row_input.uid,
                "base_prompt": row_input.base_prompt,
                "repo_urls": row_input.repo_urls,
                "scores": row_input.scores,
                "remarks": row_input.remarks,
                "results": [asdict(r) for r in repo_results],
                "analysis": {
                    "repo_issues": analysis.repo_issues,
                    "global_issues": analysis.global_issues,
                    "runtime_checks": {
                        k: asdict(v) for k, v in analysis.runtime_checks.items()
                    },
                },
                "decision": decision.model_dump(),
            }
        )

        processed += 1
        print(f"[Row {row_idx}] uid={row_input.uid}: done.")

    wb.save(output_path)
    Path(args.fetch_json).write_text(
        json.dumps(all_fetch_payloads, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Processed rows: {processed}")
    print(f"Excel written: {output_path}")
    print(f"Fetch/runtime JSON written: {args.fetch_json}")


def main() -> None:
    args = parse_args()
    process_excel(args)


if __name__ == "__main__":
    main()
