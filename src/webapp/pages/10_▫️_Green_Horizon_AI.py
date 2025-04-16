"""Green Horizon AI Chat interface for analyzing CO2 emissions data.

Provides a Streamlit-based chat interface powered by SmolAgents to query and visualize
CO2 emissions data. Includes tools for data retrieval, web search, and visualization.
"""

import folium
import pandas as pd
import streamlit as st
from devtools import debug
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    VisitWebpageTool,
    tool,
)
from streamlit_folium import st_folium

from src.ai_core.llm import LlmFactory
from src.ai_core.prompts import dedent_ws
from src.utils.config_mngr import global_config
from src.webapp.ui_components.llm_config import llm_config_widget
from src.webapp.ui_components.smoloagents_streamlit import stream_to_streamlit

MODEL_ID = None  # Use the one by configuration
# MODEL_ID = "qwen_qwq32_deepinfra"
# MODEL_ID = "gpt_o3mini_openrouter"
# MODEL_ID = "qwen_qwq32_openrouter"

SAMPLE_PROMPTS = [
    "what was the CO2 emissions of Brazil for energy generation in 2023",
    "Generate a bar chart with emissions by sector for France in 2022",
    "Create a pie chart of emissions by sector",
    "Show the change in emissions from the industrial sector between 2022 and 2024",
    "Create a simple UI with a multiselect widget and a text ",
    "Train a ML model to predict the CO2 evolution of France in the next 2 years. Display the curve with historical and predicted data",
    "Display the map of Toulouse",
]

#    "Create a UI to compare evolution of CO2 emisssions of countries (selected with a multiselect widget)",


DATASETS = {
    "country CO2 emissions": (
        "Country daily CO2 emissions from 2019 to 2024",
        "carbon_global_dataset/carbon_global.csv",
    ),
    "cities CO2 emissions": (
        "Major cities daily CO2 emissions from 2019 to 2024",
        "carbon_global_dataset/carbon_cities.csv",
    ),
    "GDP": ("Countries GDP per year", "gdp/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2.csv"),
    "country population": ("Country populatipn from 1960 to 2023", "world_population/world_population.csv"),
}


@st.cache_resource(show_spinner="Load data files")
def get_data(dataset: str) -> pd.DataFrame:
    """Load and cache CO2 emissions data from configured dataset.

    Returns:
        DataFrame containing emissions data by country and sector.
    """
    description, data_set = DATASETS[dataset]
    data_file = global_config().get_path("datasets_root") / data_set
    debug(data_file, description)
    assert data_file.exists(), f"file  not found: {data_file}"
    if data_file.name.endswith(".csv"):
        df = pd.read_csv(data_file, decimal=",")
    else:
        df = pd.read_excel(data_file)
    return df


@st.cache_resource()
def folium_map() -> folium.Map:
    return folium.Map(location=[39.949610, -75.150282], zoom_start=16)


@tool
def display_map(latitude: float, longitude: float) -> None:
    """
    Display a map at a given location (a country, a town, an address, ....)
    Args:
        latitude : latitude of the location
        longitude: longitude of the location
    """

    folium_map().location = [latitude, longitude]


@tool
def get_CO2_emissions_data_frame() -> pd.DataFrame:
    """Get CO2 emissions data frame for agent tools.

    Returns:
        DataFrame with emissions data by country and sector.
    """
    return get_data("country CO2 emissions")


st.title("Green Horizon AI Chat")
llm_config_widget(st.sidebar)

model_name = LlmFactory(llm_id=MODEL_ID).get_litellm_model_name()
llm = LiteLLMModel(model_id=model_name)

with st.expander(label="Prompt examples", expanded=True):
    st.write(SAMPLE_PROMPTS)

AUTHORIZED_IMPORTS = [
    "pandas",
    "matplotlib.pyplot",
    "matplotlib",
    "numpy",
    "json",
    "streamlit",
    "base64",
    "tempfile",
    "sklearn",
]
PRE_PROMPT = dedent_ws(f"""
    Answer following request. 
    You can use only the following packages:  {", ".join(AUTHORIZED_IMPORTS)}
    Instructions:
    - Don't generate "if __name__ == "__main__"
    - Don't use st.sidebar
    - Generate files (plot, diagrams) under temp directory.
    - Write outcomes in Markdown using 'st.markdown(...)'.\n
    - When  displaying an image, call st.image  and st.markdown with the file path. 
    - Print also the outcome on stdio, or the title if it's a diagram.
    - Use display_map function to display map
    \nRequest :
    """)

# When  displaying an image, call st.makdown with <img> tag and base64 encoded file.  Don't forget  unsafe_allow_html=True option.\n


col1, col2 = st.columns(2)
with col2:
    with st.expander("map", expanded=True):
        st_data = st_folium(folium_map(), width=725)
with col1:
    if prompt := st.chat_input("What would you like to ask SmolAgents?"):
        agent = CodeAgent(
            tools=[get_CO2_emissions_data_frame, display_map, DuckDuckGoSearchTool(), VisitWebpageTool()],
            model=llm,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
        )

        with st.container(height=600):
            stream_to_streamlit(agent, PRE_PROMPT + prompt, additional_args={"st": col2})
