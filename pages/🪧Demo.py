import streamlit as st
from pathlib import Path

def main():
    st.set_page_config(layout="wide")
    st.title("Intelligente Untertitel Demovideo")
    data_dir = Path("data")
    st.video(
        str(data_dir / "demo.mp4"),
        loop=True,
        autoplay=True,
        muted=True,
        subtitles={
            "Englisch": str(data_dir / "demo[Filtered]_en.srt"),
            "Englisch-Deutsch": str(data_dir / "demo[Filtered]_en-de.srt"),
            "Deutsch-Englisch": str(data_dir / "demo[Filtered]_de-en.srt"),
            "Deutsch": str(data_dir / "demo[Filtered]_de.srt"),
        },
    )

if __name__ == "__main__":
    main()
