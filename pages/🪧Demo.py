import streamlit as st

def main():
    st.title("Intelligente Untertitel Demovideo")
    st.video("data\\demo.mp4", loop=True, autoplay=True, muted=True, subtitles={"Englisch": "data\\demo[Filtered]_en.srt", "Englisch-Deutsch": "data\\demo[Filtered]_en-de.srt", "Deutsch-Englisch": "data\\demo[Filtered]_de-en.srt", "Deutsch": "data\\demo[Filtered]_de.srt"})

if __name__ == "__main__":
    main()
