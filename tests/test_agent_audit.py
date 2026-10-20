"""Exercise AI-response auditing through the public agent message flow."""

import hashlib
from typing import Any, AsyncGenerator, List

import pytest

from vanna import Agent, AgentConfig
from vanna.core.agent.config import AuditConfig
from vanna.core.audit import AiResponseEvent, AuditEvent, AuditLogger
from vanna.core.llm import LlmRequest, LlmResponse
from vanna.core.llm.base import LlmService
from vanna.core.llm.models import LlmStreamChunk
from vanna.core.middleware.base import LlmMiddleware
from vanna.core.registry import ToolRegistry
from vanna.core.tool import ToolCall
from vanna.core.user import User
from vanna.core.user.request_context import RequestContext
from vanna.core.user.resolver import UserResolver
from vanna.integrations.local.agent_memory import DemoAgentMemory


class RecordingAuditLogger(AuditLogger):
    def __init__(self) -> None:
        self.events: List[AuditEvent] = []

    async def log_event(self, event: AuditEvent) -> None:
        self.events.append(event)

    @property
    def responses(self) -> List[AiResponseEvent]:
        return [event for event in self.events if isinstance(event, AiResponseEvent)]


class TestUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        return User(id="analyst", email="analyst@example.com", group_memberships=[])


class ScriptedLlm(LlmService):
    def __init__(self, responses: List[LlmResponse]) -> None:
        self.responses = iter(responses)

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        return next(self.responses)

    async def stream_request(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        response = next(self.responses)
        if response.content:
            midpoint = len(response.content) // 2
            yield LlmStreamChunk(content=response.content[:midpoint])
            yield LlmStreamChunk(content=response.content[midpoint:])
        if response.tool_calls:
            yield LlmStreamChunk(tool_calls=response.tool_calls)

    async def validate_tools(self, tools: List[Any]) -> List[str]:
        return []


def make_agent(
    stream: bool,
    audit: AuditConfig,
    logger: RecordingAuditLogger | None,
    responses: List[LlmResponse],
) -> Agent:
    return Agent(
        llm_service=ScriptedLlm(responses),
        tool_registry=ToolRegistry(),
        user_resolver=TestUserResolver(),
        agent_memory=DemoAgentMemory(),
        config=AgentConfig(stream_responses=stream, audit_config=audit),
        audit_logger=logger,
    )


async def run_message(agent: Agent) -> None:
    components = [
        component
        async for component in agent.send_message(
            RequestContext(), "What is total revenue?"
        )
    ]
    assert components
    # Public send_message turns internal exceptions into error UI components.
    assert not any(
        getattr(component.rich_component, "title", None) == "Error Processing Message"
        for component in components
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("full_text", [False, True])
async def test_response_audit_honors_full_text_setting(
    stream: bool, full_text: bool
) -> None:
    logger = RecordingAuditLogger()
    text = "Total revenue is $120."
    agent = make_agent(
        stream,
        AuditConfig(include_full_ai_responses=full_text),
        logger,
        [LlmResponse(content=text)],
    )
    await run_message(agent)
    assert len(logger.responses) == 1
    event = logger.responses[0]
    assert event.response_text == (text if full_text else None)
    assert event.response_hash == hashlib.sha256(text.encode()).hexdigest()
    assert event.response_length_chars == len(text)
    assert event.user_id == "analyst"
    assert event.conversation_id
    assert event.request_id
    assert event.tool_calls_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "audit",
    [
        AuditConfig(enabled=False, include_full_ai_responses=True),
        AuditConfig(log_ai_responses=False, include_full_ai_responses=True),
    ],
)
async def test_disabled_response_audit_emits_no_event(
    stream: bool, audit: AuditConfig
) -> None:
    logger = RecordingAuditLogger()
    await run_message(
        make_agent(stream, audit, logger, [LlmResponse(content="Revenue is $120.")])
    )
    assert logger.responses == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_agent_without_audit_logger_still_completes(stream: bool) -> None:
    await run_message(
        make_agent(
            stream, AuditConfig(), None, [LlmResponse(content="Revenue is $120.")]
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_tool_only_response_and_final_answer_are_each_audited(
    stream: bool,
) -> None:
    logger = RecordingAuditLogger()
    # An unavailable tool yields a tool error, then the model can answer normally.
    tool_call = ToolCall(id="call-1", name="unavailable_sql_tool", arguments={})
    agent = make_agent(
        stream,
        AuditConfig(),
        logger,
        [
            LlmResponse(tool_calls=[tool_call]),
            LlmResponse(content="I cannot query that source."),
        ],
    )
    await run_message(agent)
    assert len(logger.responses) == 2
    first, final = logger.responses
    assert first.response_length_chars == 0
    assert first.response_hash == hashlib.sha256(b"").hexdigest()
    assert first.tool_names == ["unavailable_sql_tool"]
    assert first.tool_calls_count == 1
    assert final.tool_calls_count == 0
    assert first.request_id == final.request_id
    assert first.conversation_id == final.conversation_id


class RedactResponse(LlmMiddleware):
    async def after_llm_response(
        self, request: LlmRequest, response: LlmResponse
    ) -> LlmResponse:
        return response.model_copy(update={"content": "[redacted]"})


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_audit_uses_response_after_middleware(stream: bool) -> None:
    logger = RecordingAuditLogger()
    agent = make_agent(
        stream,
        AuditConfig(include_full_ai_responses=True),
        logger,
        [LlmResponse(content="Private synthetic answer")],
    )
    agent.llm_middlewares = [RedactResponse()]
    await run_message(agent)
    assert len(logger.responses) == 1
    assert logger.responses[0].response_text == "[redacted]"
