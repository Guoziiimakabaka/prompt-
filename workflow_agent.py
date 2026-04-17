import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

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


class AgentDecision(BaseModel):
    debug_prompt: str = Field(..., description="Debug prompt content.")
    debug_rewrite_repo: str = Field(
        ...,
        description="Selected repo id for debug, e.g. repo3 or empty string.",
    )
    new_requirement_prompt: str = Field(
        ...,
        description="New requirement prompt content.",
    )
    new_requirement_rewrite_repo: str = Field(
        ...,
        description="Selected repo id for new requirement, e.g. repo7 or empty string.",
    )
    unable_to_add_requirement_note: str = Field(
        ...,
        description="Reason note when new requirement cannot be added.",
    )


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


def _repo_exists(repo_id: str, results: list[RepoResult]) -> bool:
    return any(r.repo_id == repo_id for r in results)


def _repo_by_id(repo_id: str, results: list[RepoResult]) -> RepoResult | None:
    for r in results:
        if r.repo_id == repo_id:
            return r
    return None


def validate_and_fix_decision(
    decision: AgentDecision,
    results: list[RepoResult],
) -> AgentDecision:
    debug_repo = decision.debug_rewrite_repo.strip()
    new_repo = decision.new_requirement_rewrite_repo.strip()

    remark_candidates = [r for r in results if r.remark]
    score01 = [r for r in results if r.score in (0, 1)]
    score2 = [r for r in results if r.score == 2]

    # Debug repo guardrail: prefer remark first, then 0/1.
    if debug_repo and not _repo_exists(debug_repo, results):
        debug_repo = ""
    if debug_repo:
        selected = _repo_by_id(debug_repo, results)
        if selected is None:
            debug_repo = ""
        else:
            if selected.remark == "" and remark_candidates:
                debug_repo = remark_candidates[0].repo_id
            elif selected.score not in (0, 1) and score01:
                debug_repo = score01[0].repo_id
    else:
        if remark_candidates:
            debug_repo = remark_candidates[0].repo_id
        elif score01:
            debug_repo = score01[0].repo_id

    # New requirement guardrail: must be score=2 and exactly one repo.
    if new_repo and not _repo_exists(new_repo, results):
        new_repo = ""
    if new_repo:
        selected = _repo_by_id(new_repo, results)
        if selected is None or selected.score != 2:
            new_repo = score2[0].repo_id if score2 else ""
    else:
        new_repo = score2[0].repo_id if score2 else ""

    debug_prompt = decision.debug_prompt.strip()
    if not debug_prompt:
        if debug_repo:
            note = _repo_by_id(debug_repo, results).remark if _repo_by_id(debug_repo, results) else "存在问题"
            debug_prompt = (
                f"请修复 `{debug_repo}`，当前问题：{note}。"
                "请定位根因并修复，不要影响已有可用功能。"
            )
        else:
            debug_prompt = "7 个产物都很优秀，无法写 debug"

    new_prompt = decision.new_requirement_prompt.strip()
    unable_note = decision.unable_to_add_requirement_note.strip()
    if new_repo:
        if not new_prompt:
            new_prompt = (
                f"请基于 `{new_repo}` 新增一个具体功能：落子结果预览。"
                "要求：悬停可落子格时显示会被推动的棋子及目标位置，若将被挤出棋盘需明确标记，"
                "且点击落子后的实际结果必须和预览一致。"
            )
        unable_note = "无。已完成新增需求 repo 选择。"
    else:
        new_prompt = ""
        if not unable_note:
            unable_note = "没有可用的 2 分 case，无法新增需求。"

    return AgentDecision(
        debug_prompt=debug_prompt,
        debug_rewrite_repo=debug_repo,
        new_requirement_prompt=new_prompt,
        new_requirement_rewrite_repo=new_repo,
        unable_to_add_requirement_note=unable_note,
    )


def build_llm_input(parsed: ParsedInput, results: list[RepoResult]) -> str:
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
            }
        )

    payload = {
        "base_prompt": parsed.base_prompt,
        "rules": {
            "debug_select_priority": "优先选择有具体remark的repo；若无remark，再选0分；再选1分；debug只选1个repo",
            "new_requirement_select": "优先选择2分repo；新增需求只选1个repo",
            "new_requirement_prompt": "必须是单一具体功能，不能笼统描述",
        },
        "cases": cases,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def call_llm(parsed: ParsedInput, results: list[RepoResult]) -> AgentDecision:
    load_dotenv()
    import os

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
        "你是一个用于生成质检prompt的工程Agent。"
        "你必须基于输入case信息输出严格JSON。"
        "新增需求prompt必须是一个明确的单一功能，不允许笼统描述。"
        "debug和新增需求各只选一个repo。"
        "你输出的JSON必须包含以下键："
        "debug_prompt,debug_rewrite_repo,new_requirement_prompt,new_requirement_rewrite_repo,unable_to_add_requirement_note"
    )
    human_prompt = (
        "请根据以下输入生成结果JSON，不要输出任何解释。\n"
        f"{build_llm_input(parsed, results)}"
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    response = llm.invoke(messages)
    content = str(response.content).strip()

    # Handle optional markdown code fence.
    fence_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
    if fence_match:
        content = fence_match.group(1)

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {content}") from exc

    try:
        parsed_decision = AgentDecision.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"LLM output schema invalid: {raw}") from exc

    return validate_and_fix_decision(parsed_decision, results)


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

    decision = call_llm(parsed, results)
    write_output_md(output_path, decision)

    print(f"Generated: {output_path}")
    print(f"Fetched: {fetch_path}")


if __name__ == "__main__":
    main()
