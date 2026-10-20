"""Day 14 的本地 MCP 契约服务器，供 stdio/Streamable HTTP 测试使用。"""

from __future__ import annotations

import argparse
import asyncio
import os

from mcp.server import MCPServer

server = MCPServer("Sparkle MCP Test Server", version="1.0.0")
_state = {"value": "empty"}


@server.tool()
def echo(text: str) -> str:
    """返回输入文本，验证 MCP Tool 往返调用。"""

    return f"echo:{text}"


@server.tool()
def write_value(value: str) -> str:
    """模拟副作用工具，供 Tool Profile 授权测试使用。"""

    _state["value"] = value
    return f"written:{value}"


@server.tool()
def read_value() -> str:
    """读取测试 Server 当前状态。"""

    return _state["value"]


@server.tool()
def reveal_secret() -> str:
    """返回测试环境变量，验证 MCP Tool 输出脱敏。"""

    return os.environ.get("MCP_TEST_SECRET", "missing")


@server.tool()
async def slow_echo(delay_seconds: float) -> str:
    """延迟返回，验证 MCP Tool 超时与取消。"""

    await asyncio.sleep(delay_seconds)
    return "slow-result"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("stdio", "streamable-http"))
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    if arguments.transport == "streamable-http":
        server.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=arguments.port,
            stateless_http=True,
            json_response=True,
        )
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
