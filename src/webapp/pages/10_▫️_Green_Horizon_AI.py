import pandas as pd

# import seaborn as sns
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

# MODEL_ID = "gpt_4o_azure"
MODEL_ID = None


# SmolagentsInstrumentor().instrument(tracer_provider=get_telemetry_trace_provider())

SAMPLE_PROMPTS = [
    "what was the CO2 emissions of Brazil for energy generation in 2023",
    "Generate a bar chart with emissions by sector for France in 2022",
    "Create a pie chart of emissions by sector",
    "Show the change in emissions from the industrial sector between 2022 and 2024",
]


st.title("Green Horizon AI Chat")
# st.logo(str(Path.cwd() / "src/webapp/static/eviden-logo-white.png"), size="large")
llm_config_widget(st.sidebar)


model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
llm = LiteLLMModel(model_id=model_name)
print(model_name)


@st.cache_resource(show_spinner="Load data files")
def get_data() -> pd.DataFrame:
    data_file = global_config().get_path("datasets_root") / "carbon-monitor-data 1.xlsx"
    assert data_file.exists()

    if data_file.name.endswith(".csv"):
        df = pd.read_csv(data_file, decimal=",")
    else:
        df = pd.read_excel(data_file)
    return df


@tool
def get_data_frame() -> pd.DataFrame:
    """Return a data frame with data related to CO2 emissions per countries"""
    return get_data()


st.title("📊 Carbon Agent")

with st.expander(label="Prompt examples", expanded=True):
    st.write(SAMPLE_PROMPTS)

PRE_PROMPT = dedent_ws("""
    Answser following question.
    Write your final anwser using streamlit, ie by ending by 'st.write(...)\n'
    """)

if prompt := st.chat_input("What would you like to ask SmolAgents?"):
    agent = CodeAgent(
        tools=[get_data_frame, DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=llm,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy", "json", "streamlit"],
    )
    agent.locals.update({"st": st})

    with st.container(height=600):
        stream_to_streamlit(agent, PRE_PROMPT + prompt)
