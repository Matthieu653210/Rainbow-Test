import importlib
import importlib.util

import streamlit as st

from python.ai_core.chain_registry import find_runnable, get_runnable_registry
from python.config import get_config

st.title("💬 Runnable Playground")


RUNNABLES = {"lc_rag_example", "lc_tools_example", "lc_self_query"}
for r in RUNNABLES:
    importlib.import_module(f"python.ai_chains.{r}")

runnables_list = sorted([f"'{o.name}'" for o in get_runnable_registry()])

runnables_list = sorted([(o.tag, o.name) for o in get_runnable_registry()])
selection = st.selectbox(
    "Runnable", runnables_list, index=0, format_func=lambda x: f"[{x[0]}] {x[1]}"
)
if not selection:
    st.stop()
runnable_desc = find_runnable(selection[1])
if not runnable_desc:
    st.stop()

runnable = runnable_desc.get_runnable()

with st.expander("Runnable information", expanded=False):
    if importlib.util.find_spec("pygraphviz") is None:
        st.warning(
            "cannot draw the Runnable graph because pygraphviz and Graphviz are not installed"
        )
    else:
        drawing = runnable.get_graph().draw_png() # type: ignore
        st.image(drawing)
        st.write("")

# selected_runnable = st.selectbox("Select a Runnable", list(RUNNABLES.keys()))

with st.form("my_form"):
    input = st.text_area("Enter input:", runnable_desc.examples[0], placeholder="")
    submitted = st.form_submit_button("Submit")
    if submitted:
        llm = get_config("llm", "default_model")
        if not input:
            input = runnable_desc.examples[0]

        result = runnable_desc.invoke(input, {"llm": llm})
        st.write(result)
