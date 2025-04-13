"""Green Horizon AI Chat interface for analyzing CO2 emissions data.

Provides a Streamlit-based chat interface powered by SmolAgents to query and visualize
CO2 emissions data. Includes tools for data retrieval, web search, and visualization.
"""

import pandas as pd
import streamlit as st
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    VisitWebpageTool,
    tool,
)

from src.ai_core.llm import LlmFactory
from src.ai_core.prompts import dedent_ws
from src.utils.config_mngr import global_config
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.smoloagents_streamlit import stream_to_streamlit

MODEL_ID = None  # Use the one by configuration

SAMPLE_PROMPTS = [
    "what was the CO2 emissions of Brazil for energy generation in 2023",
    "Generate a bar chart with emissions by sector for France in 2022",
    "Create a pie chart of emissions by sector",
    "Show the change in emissions from the industrial sector between 2022 and 2024",
]


@st.cache_resource(show_spinner="Load data files")
def get_data() -> pd.DataFrame:
    """Load and cache CO2 emissions data from configured dataset.

    Returns:
        DataFrame containing emissions data by country and sector.
    """
    data_file = global_config().get_path("datasets_root") / "carbon-monitor-data 1.xlsx"
    assert data_file.exists()
    if data_file.name.endswith(".csv"):
        df = pd.read_csv(data_file, decimal=",")
    else:
        df = pd.read_excel(data_file)
    return df


@tool
def get_data_frame() -> pd.DataFrame:
    """Get CO2 emissions data frame for agent tools.

    Returns:
        DataFrame with emissions data by country and sector.
    """
    return get_data()


st.title("Green Horizon AI Chat")
llm_config_widget(st.sidebar)

model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
llm = LiteLLMModel(model_id=model_name)

with st.expander(label="Prompt examples", expanded=True):
    st.write(SAMPLE_PROMPTS)

PRE_PROMPT = dedent_ws("""
    Answer following question. 
    Use matplotlib to create plots. Use numpy for calculations.  Use sklearn to train ML models. 
    Generate files (plot, diagrams) under temp directory.
    Write outcomes in Markdown, using streamlit, ie using 'st.markdown(...)'.\n
    When  displaying an image, call st.image  and st.markdown with the file path. 
 
    Print also the outcome on stdio, or the title if it's a diagram.
    """)

# When  displaying an image, call st.makdown with <img> tag and base64 encoded file.  Don't forget  unsafe_allow_html=True option.\n

col1, col2 = st.columns(2)
with col1:
    if prompt := st.chat_input("What would you like to ask SmolAgents?"):
        agent = CodeAgent(
            tools=[get_data_frame, DuckDuckGoSearchTool(), VisitWebpageTool()],
            model=llm,
            additional_authorized_imports=[
                "pandas",
                "matplotlib.pyplot",
                "numpy",
                "json",
                "streamlit",
                "base64",
                "tempfile",
            ],
        )

        with st.container(height=600):
            stream_to_streamlit(agent, PRE_PROMPT + prompt, additional_args={"st": col2})
