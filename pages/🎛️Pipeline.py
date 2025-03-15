import streamlit as st
import os
import tempfile
import pandas as pd
from st_aggrid import AgGrid
import subprocess
from shared import play_video, generate_subtitles, process_subtitles

def main():
    st.title("Pipeline")
    
    # Initialize session state variables
    if 'subtitles_generated' not in st.session_state:
        st.session_state.subtitles_generated = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'name' not in st.session_state:
        st.session_state.name = ""
    if 'uploaded_video' not in st.session_state:
        st.session_state.uploaded_video = None
    if 'uploaded_srt_files' not in st.session_state:
        st.session_state.uploaded_srt_files = []
    # Add a session state variable for success/error messages
    if 'message' not in st.session_state:
        st.session_state.message = {"type": None, "text": None}

    option = st.radio("Wähle eine Option:", ["Lokale Dateien verwenden", "Dateien hochladen"], index=0)
    bias = st.slider("Bias", min_value=-3.0, max_value=3.0, value=0.0, step=0.1, format="%f", 
                    help="Der Bias wird vor der Anwendung der Sigma-Funktion auf die Vorhersagen angewendet. Dadurch kann die Anzahl der Untertitel nochmals angepasst werden.")
    st.markdown(f"<div style='display: flex; justify-content: space-between;'><span>Weniger Untertitel</span><span>Mehr Untertitel</span></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Display any stored messages
    if st.session_state.message["type"] == "success":
        st.success(st.session_state.message["text"])
    elif st.session_state.message["type"] == "error":
        st.error(st.session_state.message["text"])

    if option == "Lokale Dateien verwenden":
        name = st.text_input("Name des Videos (ohne Dateiendung, z.B. boysS3E4):")
        if not name:
            st.stop()
        else:
            name = "data\\" + name
            st.session_state.name = name

        audio_file = f"{name}.mp4"
        reference_file = f"{name}.srt"

        if os.path.exists(audio_file) and os.path.exists(reference_file):
            # Always show the button, regardless of whether subtitles have been generated
            if st.button("Untertitel generieren"):
                try:
                    df, srt_lines_in_memory = process_subtitles(name, audio_file, reference_file, bias)
                    st.session_state.df = df
                    success_message = "Untertitel erfolgreich generiert!"
                    st.session_state.message = {"type": "success", "text": success_message}
                    st.success(success_message)

                    df.to_csv(f"{name}.csv", index=False)
                    
                    # Generate different subtitle versions
                    generate_subtitles(name, df, srt_lines_in_memory)
                    st.session_state.subtitles_generated = True

                except Exception as e:
                    error_message = f"Fehler bei der Verarbeitung: {e}"
                    st.session_state.message = {"type": "error", "text": error_message}
                    st.error(error_message)
            
            # If subtitles have been generated, show the table and video
            if st.session_state.subtitles_generated:
                st.write("## Tabelle")
                st.write("Die Tabelle enthält die berechneten Werte nach den einzelnen Schritten. Die Spalte 'display' gibt an, ob ein Wort nach der Filterung (Schritt 3) noch angezeigt wird.")
                AgGrid(st.session_state.df, height=350, width='100%')
                
                # Play the video
                play_video(st.session_state.name)
        else:
            st.warning("Dateien nicht gefunden. Bitte stelle sicher, dass die Dateien im selben Verzeichnis wie dieses Skript liegen und den korrekten Namen haben.")
    
    else:  # Upload files option
        audio_file_upload = st.file_uploader("Lade eine Audio- oder Videodatei hoch (.mp4, .mpeg, .wav, .mp3)", 
                                           type=["mp4", "mpeg", "wav", "mp3"])
        if audio_file_upload is not None:
            st.session_state.uploaded_video = audio_file_upload
            
        reference_file_upload = st.file_uploader("Lade die Original-Untertiteldatei hoch (.srt)", 
                                               type=["srt"])
        if reference_file_upload is not None:
            st.session_state.uploaded_srt_files = [reference_file_upload]

        if st.session_state.uploaded_video is not None and len(st.session_state.uploaded_srt_files) > 0:
            name = os.path.splitext(st.session_state.uploaded_video.name)[0]
            st.session_state.name = name
            
            # Always show the button
            if st.button("Untertitel generieren"):
                # Reset file positions
                st.session_state.uploaded_video.seek(0)
                st.session_state.uploaded_srt_files[0].seek(0)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_video.name)[1]) as tmp_audio, \
                     tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:

                    # Create temporary files
                    tmp_audio.write(st.session_state.uploaded_video.read())
                    audio_file = tmp_audio.name
                    tmp_srt.write(st.session_state.uploaded_srt_files[0].read())
                    reference_file = tmp_srt.name

                    try:
                        df, srt_lines_in_memory = process_subtitles(name, audio_file, reference_file, bias)
                        st.session_state.df = df
                        success_message = "Untertitel erfolgreich generiert!"
                        st.session_state.message = {"type": "success", "text": success_message}
                        st.success(success_message)

                        # Generate different subtitle versions
                        srt_files = generate_subtitles(name, df, srt_lines_in_memory)
                        
                        # Add generated SRT files to uploaded_srt_files
                        generated_srt_files = []
                        for srt_path in srt_files.values():
                            with open(srt_path, 'rb') as f:
                                content = f.read()
                            srt_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".srt")
                            srt_file_obj.write(content)
                            srt_file_obj.close()
                            file_obj = open(srt_file_obj.name, 'rb')
                            file_obj.name = os.path.basename(srt_path)
                            generated_srt_files.append(file_obj)
                        
                        st.session_state.uploaded_srt_files.extend(generated_srt_files)
                        st.session_state.subtitles_generated = True

                    except Exception as e:
                        error_message = f"Fehler bei der Verarbeitung: {e}"
                        st.session_state.message = {"type": "error", "text": error_message}
                        st.error(error_message)
                    finally:
                        # Clean up temporary files
                        os.unlink(tmp_audio.name)
                        os.unlink(tmp_srt.name)
                        for srt_file in srt_files.values():
                            if os.path.exists(srt_file):
                                os.unlink(srt_file)
            
            # If subtitles have been generated, show the table and video
            if st.session_state.subtitles_generated:
                st.write("## Tabelle")
                st.write("Die Tabelle enthält die berechneten Werte nach den einzelnen Schritten. Die Spalte 'display' gibt an, ob ein Wort nach der Filterung (Schritt 3) noch angezeigt wird.")
                AgGrid(st.session_state.df, height=350, width='100%')
                
                # Reset file positions before playback
                st.session_state.uploaded_video.seek(0)
                for srt_file in st.session_state.uploaded_srt_files:
                    try:
                        srt_file.seek(0)
                    except:
                        pass
                
                # Play the video with uploaded files from session state
                play_video(name, st.session_state.uploaded_video, st.session_state.uploaded_srt_files)

if __name__ == "__main__":
    main()
