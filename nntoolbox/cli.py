def main():
    import sys
    from pathlib import Path
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"

    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())