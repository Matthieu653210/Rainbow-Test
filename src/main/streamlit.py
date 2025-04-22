# Main Streamlit application configuration and setup
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from src.utils.config_mngr import config_loguru, global_config

load_dotenv(verbose=True)

config_loguru()
logger.info("Starting Web Application...")

# Configure Streamlit page settings
st.set_page_config(
    page_title=global_config().get_str("ui.app_name"),
    page_icon="🛠️",
    layout="wide",  #
    initial_sidebar_state="expanded",
)


# Get Streamlit pages to display from config
pages_dir = global_config().get_dir_path("ui.pages_dir")
# Sort files by the number at the beginning of their name
pages_fn = sorted(
    pages_dir.glob("*.py"), key=lambda f: int(f.name.split("_")[0]) if f.name.split("_")[0].isdigit() else 0
)
pages = [st.Page(f.absolute()) for f in pages_fn if f.name != "__init__.py"]
pg = st.navigation(pages)
pg.run()
