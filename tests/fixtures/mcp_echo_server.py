"""Day 14 的本地 MCP 契约服务器；可由 stdio 或 Streamable HTTP 启动。"""

from __future__ import annotations

import argparse

from mcp.server import MCPServer

server = MCPServer("Sparkle MCP Contract Server", version="1.0.0")


@server.tool()
def echo(text: str) -> str:
    """返回传入文本，用于验证工具调用往返。"""

    return f"echo:{text}"


@server.tool()
def write_value(value: str) -> str:
    """模拟有副作用的工具，用于验证 MCP Tool Profile 能力声明。"""

    return f"write:{value}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    if arguments.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=arguments.port,
            stateless_http=True,
            json_response=True,
        )


if __name__ == "__main__":
    main()
