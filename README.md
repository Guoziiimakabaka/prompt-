# Prompt 质检 Agent（Excel + 行为验证驱动版）

## 目标
该 Agent 从 `input.xlsx` 逐行读取任务，针对每行的 7 个网页进行：
1. 页面内容抓取
2. Playwright 真实交互验证（行为驱动）
3. LLM 生成 `debug prompt` 与 `新增需求 prompt`
4. 结果直接回填到 Excel

---

## 输入列（必须）
- `uid`
- `prompt`
- `repo`（JSON 数组，长度必须 7）
- `评分`（JSON 数组，长度必须 7，元素只能 0/1/2）
- `产物备注（参考）`（可空，支持 `repo1: ...` 形式）

## 输出列（自动写回）
- `debug prompt`
- `debug 改写repo`
- `无法写debug改写备注`
- `新增需求 prompt`
- `新增需求改写 repo`
- `无法新增需求备注`

---

## 核心流程
1. 读取 Excel 与表头，解析输入列。
2. 逐行解析 `prompt/repo/评分/备注`。
3. 对 7 个 URL 执行静态内容抓取（标题、正文文本长度等）。
4. 对 7 个 URL 执行行为验证（Playwright）：
   - 页面加载与错误采集（pageerror / console error / requestfailed）
   - 通用交互探测（真实 click，断言 DOM 状态变化）
   - 表单探测（真实 fill，断言值变化）
   - 棋盘类探测（识别 cell/square/tile 后真实点击，断言状态变化）
5. 将行为验证结果作为主信号输入 LLM。
6. 通过后置规则强校验结果并纠偏。
7. 回填 Excel，并输出 `repo_fetch_results.json`（含行为证据）。

---

## 行为验证如何判定“有问题”
运行检查返回结构：
- `checked`
- `passed`
- `issues`
- `behavior_checked`
- `behavior_passed`
- `behavior_issues`
- `behavior_evidence`

### 触发问题的典型条件
- 页面接近空白（`#root` 未渲染且正文极短）
- 出现脚本异常或 console error
- 请求失败过多
- 执行多次真实点击后页面状态仍无变化
- 棋盘类场景中，多次点击候选落子单元后状态无变化
- 表单可见但 fill 后值未变化

说明：`behavior_evidence` 会记录探测动作和变化结果（如尝试次数、点击元素文本、状态变化次数），用于可追溯分析，而不是仅靠备注猜测。

补充：`正文提取为空` 默认作为抓取层提示信号，不会单独主导 debug 结论；仅当同时存在运行/行为异常时，才升级为可行动问题。

---

## 选取规则（已固化）
### Debug repo
优先级：
1. 有行为问题且有具体备注
2. 有行为问题且评分 0/1
3. 有具体备注
4. 0 分
5. 1 分

### 新增需求 repo
只能从满足以下条件的 repo 中选：
- 评分为 2
- `runtime.checked = true`
- `runtime.passed = true`
- `runtime.behavior_checked = true`
- `runtime.behavior_passed = true`

若不存在满足条件的 2 分 repo，则新增需求置空并填写“无法新增需求备注”。

---

## Prompt 约束（后置校验）
两个 prompt 都必须：
- 中文
- 正式、具体、非口语化
- 不包含 repo 编号、网址、英文

若 LLM 输出不合规，会自动改写；改写失败则使用内置 fallback。

---

## 运行方式
```bash
python workflow_agent.py --input_xlsx input.xlsx --sheet_name react
```

常用参数：
- `--output_xlsx output.xlsx`：输出到新文件
- `--uid 1207game-12402`：只处理指定 uid
- `--start_row 2 --max_rows 1`：分批跑
- `--fetch_json repo_fetch_results.json`：中间结果输出

---

## 依赖
- langchain
- langchain-openai
- openpyxl
- playwright
- requests
- readability-lxml
- beautifulsoup4
- python-dotenv

首次运行 Playwright 需安装浏览器：
```bash
python -m playwright install chromium
```

---

## 环境变量（.env）
- `MODEL_BASE_URL`
- `MODEL_NAME`
- `MODEL_API_KEY`

缺失任意字段会快速失败并抛异常。
