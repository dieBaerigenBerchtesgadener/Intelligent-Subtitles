import streamlit as st
import os
from shared import generate_subtitles, play_video

def main():
    st.title("Video Player")

    option = st.radio("Wähle eine Option:", ["Lokale Dateien verwenden", "Dateien hochladen"], index=0)

    if option == "Lokale Dateien verwenden":
        name = st.text_input("Name des Videos (ohne Dateiendung, z.B. boysS3E2):")
        if name:
            name = "data\\" + name
            play_video(name)

    else:
        video_file_upload = st.file_uploader("Lade eine Videodatei hoch", type=["mp4", "mpeg", "wav", "mp3"])
        srt_files_upload = st.file_uploader("Lade Untertiteldateien hoch (.srt)", type=["srt"], accept_multiple_files=True)

        if video_file_upload and srt_files_upload:
            name = os.path.splitext(video_file_upload.name)[0]
            play_video(name, video_file_upload, srt_files_upload)

if __name__ == "__main__":
    main()

