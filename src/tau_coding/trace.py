"""Tau CodingSession 事件的轻量 Trace 接缝。"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from time import time

from tau_agent.messages import WireModel
from tau_coding.events import CodingSessionEvent


class TraceEvent(WireModel):
    """一条带有顺序和 Session 关联信息的可观察事件。"""

    sequence: int
    session_id: str | None = None
    event_type: str
    timestamp: float
    event: CodingSessionEvent


class TraceRecorder:
    """记录 Tau 事件，同时原样转发给现有前端或调用方。

    Interface 的不变量是：每条输入事件恰好生成一条 Trace，序号从 1
    开始递增，capture 不丢失也不重排输入事件。
    """

    def __init__(self, *, session_id: str | None = None) -> None:
        self._session_id = session_id
        self._next_sequence = 1
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """返回当前已记录的 Trace 快照。"""

        return tuple(self._events)

    def record(self, event: CodingSessionEvent) -> TraceEvent:
        """记录一条 Tau 事件并返回对应 Trace。"""

        trace = TraceEvent(
            sequence=self._next_sequence,
            session_id=self._session_id,
            event_type=event.type,
            timestamp=time(),
            event=event,
        )
        self._events.append(trace)
        self._next_sequence += 1
        return trace

    async def capture(
        self,
        events: AsyncIterable[CodingSessionEvent],
    ) -> AsyncIterator[CodingSessionEvent]:
        """记录并原样转发一个 Tau CodingSession 事件流。"""

        async for event in events:
            self.record(event)
            yield event
