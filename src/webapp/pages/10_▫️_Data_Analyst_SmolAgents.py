"""AI Chat interface for analyzing data" """

from pathlib import Path
from typing import Any

import folium
from mcp import ToolsCapability
import pandas as pd
import streamlit as st
from devtools import debug
from groq import BaseModel
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    Tool,
    VisitWebpageTool,
    tool,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_folium import st_folium

from src.ai_core.llm import LlmFactory
from src.ai_core.prompts import dedent_ws
from src.utils.streamlit.load_data import TABULAR_FILE_FORMATS_READERS, load_tabular_data
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.smolagents_streamlit import stream_to_streamlit

MODEL_ID = None  # Use the one by configuration
# MODEL_ID = "qwen_qwq32_deepinfra"
# MODEL_ID = "gpt_o3mini_openrouter"
# MODEL_ID = "qwen_qwq32_openrouter"

DATA_PATH = Path.cwd() / "use_case_data/other"


class Demo(BaseModel):
    name: str
    source: Path | None
    tools: list[Tool] = []
    examples: list[str]


DEMOS = [
    Demo(
        name="Classic SmolAgents",
        source=None,
        examples=[
            "How many seconds would it take for a leopard at full speed to run through Pont des Arts?",
            "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?",
            "Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men’s FIFA World Cup?",
        ],
    ),
    Demo(
        name="Titanic",
        source=DATA_PATH / "titanic.csv",
        examples=[
            "What is the proportion of female passengers that survived?",
            "Were there any notable individuals or families aboard ",
            "Plot in a bar chat the proportion of male and female survivors",
            "What was the survival rate of passengers on the Titanic?",
            "Did the passenger class have an impact on survival rates?",
            "What were the ticket fares and cabin locations for the passengers?"
            "What are the demographics (age, gender, etc.) of the passengers on the Titanic?",
        ],
    ),
    Demo(
        name="CO2 Emissions",
        source=DATA_PATH / "country CO2 emissions",
        examples=[
            "what was the CO2 emissions of Brazil for energy generation in 2023",
            "Generate a bar chart with emissions by sector for France in 2022",
            "Create a pie chart of emissions by sector",
            "Show the change in emissions from the industrial sector between 2022 and 2024",
            "Create a simple UI with a multiselect widget and a text ",
            "Train a ML model to predict the CO2 evolution of France in the next 2 years. Display the curve with historical and predicted data",
        ],
    ),
    Demo(
        name="Stock Price",
        source=None,
        examples=[
            "What is the current price of Meta stock?",
            "Show me the historical prices of Apple vs Microsoft stock over the past 6 months",
        ],
    ),
    Demo(
        name="Geo",
        source=None,
        examples=["Display the map of Toulouse"],
    ),
]


@st.cache_data(show_spinner=True)
def get_dataframe(file_or_filename: Path | UploadedFile, **kwargs) -> pd.DataFrame | None:
    return load_tabular_data(file_or_filename=file_or_filename, **kwargs)

tab_demos, tab_custom = st.tabs(["Demos datasets", "Select yours"])

with tab_custom:
    raw_data_file = st.file_uploader(
        "Upload a Data file",
        type=list(TABULAR_FILE_FORMATS_READERS.keys()),
        # on_change=clear_submit,
    )
# display the list if demos from DEMOS. When one is selected, extract the tools and the 
# list of examples, and display them in 2 different columns. 
# Set the selected example in a variable "input"
# AI!
with tab_demos:
    ... 
    raw_data_file = 
    ...


df_0: pd.DataFrame | None = None
df: pd.DataFrame | None = None


with st.expander(label="Loaded Dataframe", expanded=True):
    skiprows = (
        st.number_input(
            "skip rows:",
            min_value=0,
            max_value=99,
            value=0,
            step=1,
        )
        - 1
    )
    args = {"skiprows": skiprows}
    if raw_data_file:
        file = raw_data_file
    if file:
        df = get_dataframe(file, **args)
        


#    "Create a UI to compare evolution of CO2 emisssions of countries (selected with a multiselect widget)",


# @tool
# def get_follium_map() -> folium.Map:
#     """
#     Display a map at a given location (a country, a town, an address, ....)
#     Args:
#         latitude : latitude of the location
#         longitude: longitude of the location
#     """

#     folium_map().location = [latitude, longitude]


@tool
def get_CO2_emissions_data_frame() -> pd.DataFrame:
    """Get CO2 emissions data frame for agent tools.

    Returns:
        DataFrame with emissions data by country and sector.
    """
    return get_data("country CO2 emissions")


@tool
def print_answer(answer: Any) -> None:
    """Provides a final answer to the given query

    Args:
        answer : The final answer to the query. Can be either markdown string, or a Follium.Map object, or a Path to an image, or a Pandas Dataframe.
    """

    st.session_state.agent_output.append(answer)
    print("answer displayed: {answer}")


def update_display() -> None:
    for msg in st.session_state.agent_output:
        if isinstance(msg, str):
            st.markdown(msg)
        elif isinstance(msg, folium.Map):
            st_folium(msg)
        elif isinstance(msg, pd.DataFrame):
            st.dataframe(msg)
        elif isinstance(msg, Path):
            st.image(msg)
        else:
            st.write(f"unhandled type : '{type(msg)}")


st.title("Green Horizon AI Chat")
llm_config_widget(st.sidebar)

model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
llm = LiteLLMModel(model_id=model_name)


if "agent_output" not in st.session_state:
    st.session_state.agent_output = []


with st.expander(label="Prompt examples", expanded=True):
    st.write(SAMPLE_PROMPTS)

AUTHORIZED_IMPORTS = [
    "pathlib",
    "pandas",
    "matplotlib.*",
    "numpy",
    "json",
    "streamlit",
    "base64",
    "tempfile",
    "sklearn",
    "folium",
]

FOLIUM_INSTRUCTION = dedent_ws(
    """ 
    - Use Folium to display a map. For example: 
        -- to display map at a given location, call  folium.Map([latitude, longitude])
        -- Do your best to select the zoom factor so whole location enter globaly in the map
        -- output the map object
"""
)

IMAGE_INSTRUCTION = dedent_ws(
    """ 
    -  When creating a plot or generating an image, save it as png in /temp, and call print_answer with the pathlib.Path  
"""
)

PRE_PROMPT = dedent_ws(
    f"""
    Answer following request. 
    You can use ONLY the following packages:  {", ".join(AUTHORIZED_IMPORTS)}
    Instructions:
    - Don't generate "if __name__ == "__main__"
    - Don't use st.sidebar
    - Use function 'print_answer' to generate outcome. It can be markdown, or a pathlib.Path to a generated image, or whenever possible  Python objects of Pandas Dataframe, or Follium Map.
    - Print also the outcome on stdio, or the title if it's a diagram.
    - {FOLIUM_INSTRUCTION}
    - {IMAGE_INSTRUCTION}

    \nRequest :
    """
)

# When  displaying an image, call st.makdown with <img> tag and base64 encoded file.  Don't forget  unsafe_allow_html=True option.\n


col1, col2 = st.columns(2)
with col1:
    if prompt := st.chat_input("What would you like to ask SmolAgents?"):
        # st.session_state.agent_output = []
        agent = CodeAgent(
            tools=[get_CO2_emissions_data_frame, print_answer, DuckDuckGoSearchTool(), VisitWebpageTool()],
            model=llm,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            max_steps=5,  # for debug
        )

        with st.container(height=600):
            stream_to_streamlit(agent, PRE_PROMPT + prompt, additional_args={"st": col2})

    debug(st.session_state.agent_output)
    with col2:
        st.write("answer:")
        update_display()
