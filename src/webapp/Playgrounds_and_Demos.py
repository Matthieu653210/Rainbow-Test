import os
import importlib
from pathlib import Path
from typing import Callable

import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from src.utils.config_mngr import config_loguru, global_config
from src.webapp.ui_components.llm_config import llm_config_widget

os.environ["BLUEPRINT_CONFIG"] = "edc_local"

# Load environment and config
load_dotenv(verbose=True)
config_loguru()
logger.info("Start Webapp...")
config = global_config()

# Get all page files from pages directory
PAGES_DIR = Path(__file__).parent / "pages"
page_files = [f for f in PAGES_DIR.glob("*.py") if f.name != "__init__.py"]

# Load page functions dynamically
pages = {}
for page_file in page_files:
    module_name = f"src.webapp.pages.{page_file.stem}"
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "main"):
            pages[page_file.stem] = module.main
    except Exception as e:
        logger.error(f"Error loading page {page_file.stem}: {e}")

# Sidebar configuration
logo_eviden = str(Path.cwd() / "src/webapp/static/eviden-logo-white.png")
st.sidebar.image(logo_eviden, width=120)
llm_config_widget(st.sidebar)

# Navigation
selected_page = st.sidebar.radio(
    "Select a demo",
    list(pages.keys()),
    format_func=lambda x: x.replace("_", " ").title()
)

# Main content
if selected_page in pages:
    pages[selected_page]()
else:
    st.error("Selected page not found")
