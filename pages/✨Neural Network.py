import streamlit as st
import pandas as pd
from model import predict_with_bias
from utils import device
import os
import torch
from types import SimpleNamespace
import wandb
from subtitle_generation import create_srt_file
from preprocessing import read_srt_in_memory, extract_tokens_with_sentences
import tempfile
from shared import generate_subtitles, play_video
from utils import cefr_levels  # Import the CEFR mapping

def main():
    st.set_page_config(layout="wide")
    st.title("Neural Network")

    option = st.radio("Wähle eine Option:", ["Lokale Datei verwenden", "CSV-Datei hochladen"], index=0)

    if option == "Lokale Datei verwenden":
        name = st.text_input("Name der CSV-Datei (ohne Dateiendung, z.B. boysS3E2):")
        if not name:
            st.stop()
        else:
            name = "data\\" + name

        csv_file = f"{name}.csv"
        original_srt = f"{name}.srt"

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            bias = st.slider("Bias", min_value=-3.0, max_value=3.0, value=0.0, step=0.1, format="%f", 
                             help="Der Bias wird vor der Anwendung der Sigma-Funktion auf die Vorhersagen angewendet. Dadurch kann die Anzahl der Untertitel nochmals angepasst werden.")
            st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Weniger Untertitel</span><span>Mehr Untertitel</span></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])

            with col1:
                language_level = st.selectbox(
                    "English level:",
                    options=list(cefr_levels.keys()),
                    index=None,
                    placeholder="Select your English level",
                    key='language_level_select',
                    help="Not sure about your level? Take a test: https://test-english.com/level-test/"
                )
            
            with col2:
                audio_level = st.selectbox(
                    "Audio level:",
                    options=list(cefr_levels.keys()),
                    index=None,
                    placeholder="Select your Audio level",
                    key='audio_level_select',
                    help="Not sure about your listening level? Take a test: https://www.oxfordonlineenglish.com/english-level-test/listening"
                )

            if st.button("Untertitel generieren"):
                srt_lines_in_memory = read_srt_in_memory(original_srt)
                df['audio_level'] = cefr_levels[audio_level]
                df['language_level'] = cefr_levels[language_level]
                df.loc[~df['set_manually'], 'display'] = predict_with_bias(df.loc[~df['set_manually']], device, bias=bias)
                df.to_csv(csv_file, index=False)
                
                # Generiere Untertitel in verschiedenen Sprachen
                srt_files = generate_subtitles(name, df, srt_lines_in_memory, english_level=(cefr_levels[language_level]-0.2))
                st.success("Untertitel erfolgreich generiert!")

            # Video mit Untertiteln anzeigen
            play_video(name)

        else:
            st.error(f"Datei {name}.csv nicht gefunden.")

    else:
        uploaded_file = st.file_uploader("Lade eine CSV-Datei hoch", type=["csv"])
        video_file_upload = st.file_uploader("Lade optional eine Videodatei hoch", type=["mp4", "mpeg", "wav", "mp3"])
        original_srt_upload = st.file_uploader("Lade die Original-Untertiteldatei hoch (.srt)", type=["srt"])

        if uploaded_file is not None and original_srt_upload is not None:
            df = pd.read_csv(uploaded_file)
            bias = st.slider("Bias", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)

            if st.button("Untertitel generieren"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
                    tmp_srt.write(original_srt_upload.read())
                    original_srt = tmp_srt.name
                    srt_lines_in_memory = read_srt_in_memory(original_srt)
                    
                    # Vorhersagen generieren
                    df.loc[~df['set_manually'], 'display'] = predict_with_bias(df.loc[~df['set_manually']], device, bias=bias)
                    
                    # Temporärer Name für die Untertitel
                    temp_name = "temp_video"
                    
                    # Generiere Untertitel in verschiedenen Sprachen
                    srt_files = generate_subtitles(temp_name, df, srt_lines_in_memory)
                    st.success("Untertitel erfolgreich generiert!")

                    # Video mit Untertiteln anzeigen wenn vorhanden
                    if video_file_upload:
                        play_video(temp_name, video_file_upload, [original_srt_upload] + 
                                 [open(srt_file, 'rb') for srt_file in srt_files.values()])

                    # Aufräumen
                    os.unlink(tmp_srt.name)
                    for srt_file in srt_files.values():
                        if os.path.exists(srt_file):
                            os.unlink(srt_file)

if __name__ == "__main__":
    main()
