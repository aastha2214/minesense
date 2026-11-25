"""MineSense demo app.

    streamlit run app/streamlit_app.py

Pages: a single-reading classifier with an uncertainty flag, a minefield
sweep simulation, and the cross-validated results.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
st.set_page_config(page_title="MineSense", page_icon="🧭", layout="wide")
import views  # noqa: E402

pages = [st.Page(views.single_reading, title="Single reading", icon="📟", default=True),
         st.Page(views.minefield, title="Minefield simulation", icon="🗺️"),
         st.Page(views.results, title="Results", icon="📊")]
st.navigation(pages).run()
