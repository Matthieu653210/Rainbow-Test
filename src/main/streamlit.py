# Main Streamlit application configuration and setup
import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from src.utils.config_mngr import config_loguru, global_config

# Load environment variables from .env file
load_dotenv(verbose=True)

# Configure logging using Loguru
config_loguru()
logger.info("Starting Web Application...")

# Configure Streamlit page settings
st.set_page_config(
    page_title=global_config().get_str("ui.app_name"),  # Get app name from config
    page_icon="🛠️",  # Tool emoji as icon
    layout="wide",  # Use wide layout
    initial_sidebar_state="expanded",  # Start with sidebar expanded
)


pages_dir = global_config().get_path("ui.pages_dir")
# Sort files by the number at the beginning of their name
pages_fn = sorted(
    pages_dir.glob("*.py"), key=lambda f: int(f.name.split("_")[0]) if f.name.split("_")[0].isdigit() else 0
)
pages = [st.Page(f.absolute()) for f in pages_fn if f.name != "__init__.py"]
pg = st.navigation(pages)
pg.run()


# def main() -> None:
#     # taken from https://blog.yericchen.com/python/installable-streamlit-app.html
#     # Does not work as expected
#     script_path = __file__
#     import subprocess
#     subprocess.run(["streamlit", "run", script_path])
