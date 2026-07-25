# -*- coding: utf-8 -*-
"""
Agent Executor — ReAct loop with tool calling.

Orchestrates the LLM + tools interaction loop:
1. Build system prompt (persona + tools + skills)
2. Send to LLM with tool declarations
3. If tool_call → execute tool → feed result back
4. If text → parse as final answer
5. Loop until final answer or max_steps

The core execution loop is delegated to :mod:`src.agent.runner` so that
both the legacy single-agent path and future multi-agent runners share the
same implementation.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.config import get_config
from src.agent.chat_context import build_agent_chat_context_bundle, build_visible_chat_history
from src.agent.llm_adapter import LLMToolAdapter
from src.agent.provider_trace import persist_provider_trace_turns
from src.agent.runner import run_agent_loop, parse_dashboard_json
from src.agent.runtime_facts import AgentRuntimeFacts
from src.agent.stock_scope import StockScope, resolve_stock_scope
from src.storage import get_db
from src.agent.tools.registry import ToolRegistry
from src.report_language import normalize_report_language
from src.market_context import get_market_role, get_market_guidelines
from src.market_phase_prompt import format_market_phase_prompt_section
from src.market_structure_prompt import format_market_structure_prompt_section
from src.services.daily_market_context import format_daily_market_context_prompt_section

logger = logging.getLogger(__name__)


# ============================================================
# Agent result
# ============================================================

@dataclass
class AgentResult:
    """Result from an agent execution run."""
    success: bool = False
    content: str = ""                          # final text answer from agent
    dashboard: Optional[Dict[str, Any]] = None  # parsed dashboard JSON
    tool_calls_log: List[Dict[str, Any]] = field(default_factory=list)  # execution trace
    total_steps: int = 0
    total_tokens: int = 0
    provider: str = ""
    model: str = ""                            # comma-separated models used (supports fallback)
    error: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    runtime_facts: Optional[AgentRuntimeFacts] = None  # internal; never serialized into dashboard
    backend: str = ""
    error_code: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


# ============================================================
# System prompt builder
# ============================================================

LEGACY_DEFAULT_AGENT_SYSTEM_PROMPT = """你是一位专注于趋势交易的{market_role}投资分析 Agent，拥有数据工具和交易技能，负责生成专业的【决策仪表盘】分析报告。

{market_guidelines}

## 工作流程（必须严格按阶段顺序执行，每阶段等工具结果返回后再进入下一阶段）

**第一阶段 · 行情与K线**（首先执行）
- `get_realtime_quote` 获取实时行情
- `get_daily_history` 获取历史K线

**第二阶段 · 技术与筹码**（等第一阶段结果返回后执行）
- `analyze_trend` 获取技术指标
- `get_chip_distribution` 获取筹码分布

**第三阶段 · 情报搜索**（等前两阶段完成后执行）
- `search_stock_news` 搜索最新资讯、减持、业绩预告等风险信号

**第四阶段 · 生成报告**（所有数据就绪后，输出完整决策仪表盘 JSON）

> ⚠️ 每阶段的工具调用必须完整返回结果后，才能进入下一阶段。禁止将不同阶段的工具合并到同一次调用中。
{default_skill_policy_section}

## 规则

1. **必须调用工具获取真实数据** — 绝不编造数字，所有数据必须来自工具返回结果。
2. **系统化分析** — 严格按工作流程分阶段执行，每阶段完整返回后再进入下一阶段，**禁止**将不同阶段的工具合并到同一次调用中。
3. **应用交易技能** — 评估每个激活技能的条件，在报告中体现技能判断结果。
4. **输出格式** — 最终响应必须是有效的决策仪表盘 JSON。
5. **风险优先** — 必须排查风险（股东减持、业绩预警、监管问题）。
6. **工具失败处理** — 记录失败原因，使用已有数据继续分析，不重复调用失败工具。

{skills_section}

## 输出格式：决策仪表盘 JSON

你的最终响应必须是以下结构的有效 JSON 对象：

```json
{{
    "stock_name": "股票中文名称",
    "sentiment_score": 0-100整数,
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "decision_type": "buy/hold/sell",
    "confidence_level": "高/中/低",
    "dashboard": {{
        "core_conclusion": {{
            "one_sentence": "一句话核心结论（30字以内）",
            "signal_type": "🟢买入信号/🟡持有观望/🔴卖出信号/⚠️风险警告",
            "time_sensitivity": "立即行动/今日内/本周内/不急",
            "position_advice": {{
                "no_position": "空仓者建议",
                "has_position": "持仓者建议"
            }}
        }},
        "data_perspective": {{
            "trend_status": {{"ma_alignment": "", "is_bullish": true, "trend_score": 0}},
            "price_position": {{"current_price": 0, "ma5": 0, "ma10": 0, "ma20": 0, "bias_ma5": 0, "bias_status": "", "support_level": 0, "resistance_level": 0}},
            "volume_analysis": {{"volume_ratio": 0, "volume_status": "", "turnover_rate": 0, "volume_meaning": ""}},
            "chip_structure": {{"profit_ratio": 0, "avg_cost": 0, "concentration": 0, "chip_health": ""}}
        }},
        "intelligence": {{
            "latest_news": "",
            "risk_alerts": [],
            "positive_catalysts": [],
            "earnings_outlook": "",
            "sentiment_summary": ""
        }},
        "battle_plan": {{
            "sniper_points": {{"ideal_buy": "", "secondary_buy": "", "stop_loss": "", "take_profit": ""}},
            "position_strategy": {{"suggested_position": "", "entry_plan": "", "risk_control": ""}},
            "action_checklist": []
        }},
        "phase_decision": {{
            "phase_context": {{"phase": "premarket/intraday/lunch_break/closing_auction/postmarket/non_trading/unknown"}},
            "action_window": "盘前计划/盘中跟踪/午间确认/收盘前风控/盘后复盘/非交易日观察",
            "immediate_action": "立即行动/等待确认/观察/止损止盈预警/禁止追高/无盘中动作",
            "watch_conditions": ["观察条件1", "观察条件2"],
            "next_check_time": "下一次检查点或市场本地时间",
            "confidence_reason": "置信度理由，说明阶段和数据质量限制",
            "data_limitations": ["阶段或数据质量限制1", "阶段或数据质量限制2"]
        }},
        "signal_attribution": {{
            "technical_indicators": 技术指标贡献度(0-100；有效非零贡献度之和应为100；全零表示无有效信号),
            "news_sentiment": 新闻舆情贡献度(0-100；有效非零贡献度之和应为100；全零表示无有效信号),
            "fundamentals": 基本面贡献度(0-100；有效非零贡献度之和应为100；全零表示无有效信号),
            "market_conditions": 市场环境贡献度(0-100；有效非零贡献度之和应为100；全零表示无有效信号),
            "strongest_bullish_signal": "最强看多信号名称",
            "strongest_bearish_signal": "最强看空信号名称"
        }}
    }},
    "analysis_summary": "100字综合分析摘要",
    "key_points": "3-5个核心看点，逗号分隔",
    "risk_warning": "风险提示",
    "buy_reason": "操作理由，引用交易理念",
    "trend_analysis": "走势形态分析",
    "short_term_outlook": "短期1-3日展望",
    "medium_term_outlook": "中期1-2周展望",
    "technical_analysis": "技术面综合分析",
    "ma_analysis": "均线系统分析",
    "volume_analysis": "量能分析",
    "pattern_analysis": "K线形态分析",
    "fundamental_analysis": "基本面分析",
    "sector_position": "板块行业分析",
    "company_highlights": "公司亮点/风险",
    "news_summary": "新闻摘要",
    "market_sentiment": "市场情绪",
    "hot_topics": "相关热点"
}}
