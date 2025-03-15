import streamlit as st
import torch
from streamlit_pdf_viewer import pdf_viewer
import reveal_slides as rs
import streamlit.components.v1 as components

# Fix for error
torch.classes.__path__ = []

st.set_page_config(
    page_title="Intelligente Untertitel",
    page_icon="🎬",
    layout="wide"
)

def main():
    st.title("Intelligente Untertitel")
    st.video("data\\1.mp4", start_time=29, end_time=60)
    st.video("data\\1.mp4", start_time=29, end_time=60, subtitles="data\\1.srt")

    #with open("presentation.html", "r") as file:
    #    content_markup = file.read()
    #components.html(content_markup, height=600, scrolling=True)

    pdf_viewer("Jugend forscht Arbeit Landeswettbewerb.pdf", width="90%", height=2000)


if __name__ == "__main__":
    main()
