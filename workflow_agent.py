import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, HttpUrl, ValidationError


class AgentInput(BaseModel):
    original_prompt: str = Field(..., min_length=1)
    repo: list[HttpUrl] = Field(..., min_length=7, max_length=7)
    scores: list[int] = Field(..., min_length=7, max_length=7)
    score_notes: list[str] = Field(..., min_length=7, max_length=7)
    manual_check: list[bool] = Field(
        ...,
        min_length=7,
        max_length=7,
        description="True means bug exists after manual check, False means no bug.",
    )


class AgentOutput(BaseModel):
    debug_prompt: str
    debug_rewrite_repo: str
    new_requirement_prompt: str
    new_requirement_rewrite_repo: str
    unable_to_add_requirement_note: str


@dataclass
class CaseRecord:
    case_id: str
    url: str
    score: int
    score_note: str
    has_bug_after_manual_check: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-workflow agent for debug prompt and new requirement prompt."
    )
    parser.add_argument(
        "--input_json",
        default="",
        help="Input JSON string with keys: original_prompt, repo, scores, score_notes, manual_check",
    )
    parser.add_argument(
        "--input_file",
        default="",
        help="Path to input JSON file. If provided, this takes precedence over --input_json.",
    )
    parser.add_argument(
        "--output_file",
        default="agent_output.json",
        help="Output JSON file path.",
    )
    return parser.parse_args()


def build_case_records(payload: AgentInput) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    for idx in range(7):
        case = CaseRecord(
            case_id=f"repo{idx + 1}",
            url=str(payload.repo[idx]),
            score=payload.scores[idx],
            score_note=payload.score_notes[idx],
            has_bug_after_manual_check=payload.manual_check[idx],
        )
        cases.append(case)
    return cases


def detect_prompt_abnormal(original_prompt: str) -> str:
    text = original_prompt.strip()
    if len(text) < 30:
        return "原始需求可能过短，信息不足，存在需求缺失风险。"

    abnormal_markers = [
        "以下是一个游戏设计",
        "从零开始",
    ]
    marker_hits = sum(1 for marker in abnormal_markers if marker in text)
    if marker_hits == 0:
        return "原始需求可能存在表述模糊，请人工复核需求完整性。"
    return ""


def choose_debug_case(cases: list[CaseRecord]) -> CaseRecord | None:
    score_zero = [case for case in cases if case.score == 0]
    if score_zero:
        return score_zero[0]

    score_one = [case for case in cases if case.score == 1]
    if score_one:
        return score_one[0]

    return None


def choose_new_requirement_case(cases: list[CaseRecord]) -> CaseRecord | None:
    valid = [
        case
        for case in cases
        if case.score == 2 and case.has_bug_after_manual_check is False
    ]
    if not valid:
        return None
    return valid[0]


def build_debug_prompt_without_llm(case: CaseRecord) -> str:
    return (
        f"请帮我重点修复 {case.case_id} 这个静态网站。"
        f"我实际打开后发现问题是：{case.score_note}。"
        "请你先定位根因，再把问题修好。"
        "修复要求：不要改坏已有可用功能，修完后请自查核心交互流程。"
    )


def build_new_requirement_prompt_without_llm(
    original_prompt: str,
    case: CaseRecord,
) -> str:
    return (
        f"当前 {case.case_id} 已通过检查，没有发现 bug。"
        "请在不偏离原始需求的前提下继续增强功能，新增需求必须与当前产品核心场景强相关。"
        "增强要求：保持现有交互逻辑一致，界面风格延续当前版本，新增功能完成后可直接在页面中体验。"
        f"原始需求摘要：{original_prompt[:180]}"
    )


def create_llm_client() -> OpenAI:
    base_url = os.getenv("MODEL_BASE_URL")
    api_key = os.getenv("MODEL_API_KEY")

    if not base_url:
        raise ValueError("MODEL_BASE_URL is missing in .env")
    if not api_key:
        raise ValueError("MODEL_API_KEY is missing in .env")

    return OpenAI(base_url=base_url, api_key=api_key)


def llm_rewrite_prompt(
    client: OpenAI,
    model_name: str,
    prompt_type: str,
    draft_prompt: str,
    original_prompt: str,
    case: CaseRecord,
    abnormal_note: str,
) -> str:
    system_prompt = (
        "你是一个严谨的提示词优化助手。"
        "请只输出最终中文提示词内容，不要输出解释。"
    )

    user_prompt = (
        f"任务类型：{prompt_type}\n"
        f"case_id：{case.case_id}\n"
        f"case_url：{case.url}\n"
        f"case_score：{case.score}\n"
        f"case_note：{case.score_note}\n"
        f"原始需求：{original_prompt}\n"
        f"异常备注：{abnormal_note or '无'}\n"
        f"初稿：{draft_prompt}\n\n"
        "请将初稿改写为更口语化、具体、可执行的提示词。"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    content = response.choices[0].message.content
    if content is None or not content.strip():
        raise ValueError("LLM returned empty content.")
    return content.strip()


def run_workflow(input_payload: dict[str, Any]) -> AgentOutput:
    payload = AgentInput.model_validate(input_payload)
    for score in payload.scores:
        if score not in (0, 1, 2):
            raise ValueError("scores only allow 0/1/2")

    cases = build_case_records(payload)
    abnormal_note = detect_prompt_abnormal(payload.original_prompt)

    debug_case = choose_debug_case(cases)
    new_case = choose_new_requirement_case(cases)

    load_dotenv()
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        raise ValueError("MODEL_NAME is missing in .env")

    client = create_llm_client()

    if debug_case is None:
        debug_prompt = "7 个产物都很优秀，无法写 debug"
        debug_repo = ""
    else:
        debug_draft = build_debug_prompt_without_llm(debug_case)
        debug_prompt = llm_rewrite_prompt(
            client=client,
            model_name=model_name,
            prompt_type="debug",
            draft_prompt=debug_draft,
            original_prompt=payload.original_prompt,
            case=debug_case,
            abnormal_note=abnormal_note,
        )
        debug_repo = debug_case.case_id

    if new_case is None:
        new_requirement_prompt = ""
        new_requirement_repo = ""
        score2_cases = [case for case in cases if case.score == 2]
        if not score2_cases:
            unable_note = "没有可用的 2 分 case，无法写新增需求。"
        else:
            unable_note = "所有 2 分 case 在实际检查后仍存在 bug，无法新增需求。"
    else:
        new_draft = build_new_requirement_prompt_without_llm(
            original_prompt=payload.original_prompt,
            case=new_case,
        )
        new_requirement_prompt = llm_rewrite_prompt(
            client=client,
            model_name=model_name,
            prompt_type="new_requirement",
            draft_prompt=new_draft,
            original_prompt=payload.original_prompt,
            case=new_case,
            abnormal_note=abnormal_note,
        )
        new_requirement_repo = new_case.case_id
        unable_note = ""

    if abnormal_note:
        suffix = (
            " 原始需求疑似异常，请人工复核。"
            "若异常影响理解，请在对应 case 标注异常来源。"
        )
        if unable_note:
            unable_note = f"{unable_note} {suffix}"
        else:
            unable_note = suffix

    return AgentOutput(
        debug_prompt=debug_prompt,
        debug_rewrite_repo=debug_repo,
        new_requirement_prompt=new_requirement_prompt,
        new_requirement_rewrite_repo=new_requirement_repo,
        unable_to_add_requirement_note=unable_note,
    )


def main() -> None:
    args = parse_args()
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8-sig") as file:
            input_payload = json.load(file)
    else:
        if not args.input_json:
            raise ValueError("Either --input_file or --input_json must be provided.")
        try:
            input_payload = json.loads(args.input_json)
        except json.JSONDecodeError as exc:
            raise ValueError("input_json is not valid JSON string") from exc

    result = run_workflow(input_payload)
    with open(args.output_file, "w", encoding="utf-8") as file:
        json.dump(result.model_dump(), file, ensure_ascii=False, indent=2)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
