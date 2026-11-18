"""
Streamlit Cloud Entry Point wrapper
Enables automatic discovery whether the user specifies 'streamlit_app.py' or 'app.py'.
"""
import runpy
from pathlib import Path

app_path = Path(__file__).parent / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
