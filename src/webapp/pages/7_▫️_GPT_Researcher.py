""" """

import asyncio
import textwrap
from collections import deque
from datetime import datetime
from typing import Any, Final

import pandas as pd
import streamlit as st
from langchain.callbacks import tracing_v2_enabled

from src.ai_core.llm import configurable
from src.ai_extra.gpt_researcher_chain import (
    GptrConfVariables,
    ReportType,
    SearchEngine,
    Tone,
    gpt_researcher_chain,
)

LOG_SIZE_MAX = 100

# GPTR_LLM_ID = "gpt_4omini_openrouter"
GPTR_LLM_ID = "deepseek:deepseek-chat"

CUSTOM_GPTR_CONFIG = {
    "MAX_ITERATIONS": 3,
    "MAX_SEARCH_RESULTS_PER_QUERY": 5,
}

st.title("GPT Researcher Playground")

with st.sidebar:
    st.write("hello")

SAMPLE_SEARCH = [
    "What are the ethical issues with AI autonomous agents ? ",
    "What is the architecture of SmolAgents and how it compare with LangGraph ? ",
    "What are the Agentic AI  solutions announced by AWS, Google, Microsoft, SalesForce, Service Now, UI Path, SAP, and other major software editors",
]

# See https://docs.gptr.dev/docs/gpt-researcher/gptr/config
#

with st.expander(label="Search Configuration"):
    col1, col2, col3 = st.columns(3)
    col1.number_input("Max Interation", 1, 5, CUSTOM_GPTR_CONFIG["MAX_ITERATIONS"])
    col1.number_input("Max search per query", 1, 10, CUSTOM_GPTR_CONFIG["MAX_SEARCH_RESULTS_PER_QUERY"])
    search_mode = col2.selectbox("Search Mode", [rt.value for rt in ReportType])
    col2.selectbox("Search Engine", [rt.value for rt in SearchEngine])
    col2.selectbox("Tone", [rt.value for rt in Tone])
    if search_mode == "custom_report":
        col3.text_area("System prompt:", height=150)
    st.write("Not Yet Implemented".upper())


# data = sp.pydantic_form(key="Configuration", model=CommonConfigParams)

col1, co2 = st.columns([4, 1])
sample_search = col1.selectbox("Sample queries", SAMPLE_SEARCH, index=None)

search_input = col1.text_area("Your query", height=70, placeholder=" query here...", value=sample_search)
use_cached_result = co2.checkbox("Use cache", value=True, help="Use previous cached search and analysis outcomes ")


if "log_entries" not in st.session_state:
    st.session_state.log_entries = deque(maxlen=100)
if "research_full_report" not in st.session_state:
    st.session_state.research_full_report = None

if "traces" not in st.session_state:
    st.session_state.traces = {}


class CustomLogsHandler:
    """Handles real-time logging display in Streamlit UI

    Manages both synchronous and asynchronous logging to a Streamlit container.
    Maintains a circular buffer of log entries for display and history.

    Attributes:
        log_container: Streamlit container element for displaying logs
    """

    def __init__(self, log_container, height=200) -> None:
        self.log_container = log_container.container(height=height)

    async def send_json(self, data: dict[str, Any]) -> None:
        if "log_entries" not in st.session_state:
            st.session_state.log_entries = deque(maxlen=100)
        data["ts"] = datetime.now().strftime("%H:%M:%S")
        st.session_state.log_entries.append(data)

        with self.log_container:
            # Automatically scroll to show latest log entries
            st.markdown(
                """
                <style>
                .stContainer {
                    overflow-y: auto;
                    max-height: 200px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            def stream_log():
                for entry in st.session_state["log_entries"]:
                    line = textwrap.shorten(entry["output"], 120)
                    yield f"{line}\n"

            st.write_stream(stream_log)

    async def write_log(self, line: str) -> None:
        """Write a single log line to the streamlit container"""
        await self.send_json({"output": line})


# log_container = None


researcher_conf = GptrConfVariables(
    # fast_llm_id=gpt_llm,
    # smart_llm_id=gpt_llm,
    # strategic_llm_id=gpt_llm,
    extra_params=CUSTOM_GPTR_CONFIG
)


async def main() -> None:
    """Main async function handling the Streamlit UI and search operations

    Manages:
    - UI layout and state initialization
    - LLM search execution and results display
    - Web research execution and comprehensive reporting
    - Traceability and debugging support

    UI Flow:
    1. User inputs question
    2. Chooses between LLM or Web search
    3. Results displayed in organized tabs:
       - LLM Search: Breakdowns, synthesis, stats
       - Web Search: Report, context, images, sources
    """

    with st.form("my_form"):
        submitted_web_search = st.form_submit_button("Web Search", disabled=search_input is None)

        if submitted_web_search and search_input:
            log, report_tab, context_tab, image_tab, sources_tab, stats_tab_web = st.tabs(
                ["log", "**Report**", "Context", "Images", "Sources", "Stats"]
            )
            log_handler = CustomLogsHandler(log, 200)

            gptr_params = {"report_source": "web", "tone": "Objective"}
            gptr_chain = gpt_researcher_chain().with_config(
                configurable(
                    {
                        "logger": log_handler,
                        "gptr_conf": researcher_conf,
                        "gptr_params": gptr_params,
                        "use_cached_result": True,
                    }
                )
            )
            with tracing_v2_enabled() as cb:
                with st.spinner(text="searching the web..."):
                    st.session_state.research_full_report = await gptr_chain.ainvoke(search_input)
                    st.session_state.traces["web_search"] = cb.get_run_url()
                    await log_handler.write_log("The search report ready !")

            research_full_report = st.session_state.research_full_report
            if research_full_report:
                # write in fist tabs
                web_research_result = research_full_report.report
                report_tab.write(web_research_result)
                context_tab.write(research_full_report.context)

                # 'Image' tab content
                nb_col: Final = 4
                image_tab.write(f"Found images (len: {len(research_full_report.images)})")
                image_cols = image_tab.columns(nb_col)
                for index, image_path in enumerate(research_full_report.images):
                    with image_cols[index % nb_col]:
                        try:
                            st.image(image_path, width=200, caption=f"Image {index + 1}", use_container_width=False)
                        except Exception:
                            st.write(f"cannot display {image_path}")

                # 'Source' tab content
                source_dict = {(s["url"], s["title"]) for s in research_full_report.sources}
                df = pd.DataFrame(source_dict, columns=["url", "title"], index=None)
                sources_tab.dataframe(df)

                # 'Stats' tab content
                stats_tab_web.write(f"Research costs: ${research_full_report.costs}")
                if trace_url := st.session_state.traces.get("web_search"):
                    stats_tab_web.write(f"trace: {trace_url}")


asyncio.run(main())
