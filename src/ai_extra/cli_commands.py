import asyncio
import sys
from typing import Annotated, Optional

import typer
from langchain.globals import set_debug, set_verbose
from smolagents import (
    CodeAgent,
)
from smolagents.default_tools import TOOL_MAPPING

# Import modules where runnables are registered
from typer import Option

from src.ai_core.cache import LlmCache
from src.ai_core.llm import LlmFactory
from src.ai_extra.mcp_client import call_react_agent
from src.utils.config_mngr import global_config


def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def mcp_agent(
        input: str | None = None,
        cache: str = "memory",
        lc_verbose: Annotated[bool, Option("--verbose", "-v")] = False,
        lc_debug: Annotated[bool, Option("--debug", "-d")] = False,
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
    ) -> None:
        """
        Quick test
        """
        set_debug(lc_debug)
        set_verbose(lc_verbose)
        LlmCache.set_method(cache)

        if llm_id is not None:
            if llm_id not in LlmFactory.known_items():
                print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
                return
            global_config().set("llm.default_model", llm_id)

        if not input and not sys.stdin.isatty():
            input = sys.stdin.read()
        if not input or len(input) < 5:
            print("Error: Input parameter or something in stdin is required")
            return

        asyncio.run(call_react_agent(input))

    @cli_app.command()
    def smolagents(
        prompt: str,
        tools: Annotated[list[str], Option("--tools", "-t")] = [],
        llm_id: Annotated[Optional[str], Option("--llm-id", "-m")] = None,
        imports: list[str] | None = None,
    ) -> None:
        """
        ex: "How many seconds would it take for a leopard at full speed to run through Pont des Arts?" -t web_search
        """
        if llm_id is not None:
            if llm_id not in LlmFactory.known_items():
                print(f"Error: {llm_id} is unknown llm_id.\nShould be in {LlmFactory.known_items()}")
                return
            global_config().set("llm.default_model", llm_id)

        model = LlmFactory(llm_id=llm_id).get_smolagent_model()
        available_tools = []
        for tool_name in tools:
            if "/" in tool_name:
                available_tools.append(Tool.from_space(tool_name))
            else:
                if tool_name in TOOL_MAPPING:
                    available_tools.append(TOOL_MAPPING[tool_name]())
                else:
                    raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

        print(f"Running agent with these tools: {tools}")
        agent = CodeAgent(tools=available_tools, model=model, additional_authorized_imports=imports)

        agent.run(prompt)
