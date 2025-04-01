from subtitle_generation import create_srt_file
import os
import streamlit as st
import tempfile
from preprocessing import read_srt_in_memory, extract_tokens_with_sentences, isolate_speech
from audio_complexity import extract_complexity, improve_timesteps
from filter_in_out import mark_non_english_in_df, mark_notes_in_df, mark_excluded_words, mark_numbers_in_df
from translation import batch_translate_and_align
from feature_extraction import FeatureExtractor
from model import prepare_data, evaluate_model, predict_with_bias, start_training, BinaryClassifier
from subtitle_generation import create_srt_file
from utils import device
import time
import pandas as pd
import wandb
from types import SimpleNamespace
import torch
import streamlit.components.v1 as components
import base64
from pydub import AudioSegment
import io
import re

def process_subtitles(name, audio_file, reference_file, bias=0.0, predict=True, audio_level=0.0, language_level=0.0):
    """
    Hauptfunktion zum Ausführen der Untertitel-Pipeline mit Status-Updates und DataFrame-Anzeige.
    """
    # 1. Daten einlesen und vorverarbeiten
    with st.status("Lade SRT-Datei und extrahiere Tokens...", expanded=True) as status:
        start_time = time.time()
        srt_lines_in_memory = read_srt_in_memory(reference_file)
        original_tokens = extract_tokens_with_sentences(srt_lines_in_memory)
        end_time = time.time()
        runtime = end_time - start_time
        status.update(label=f"SRT-Datei erfolgreich geladen und Tokens extrahiert in {runtime:.2f} Sekunden", state="complete", expanded=False)

    # 2. Audiokomplexität berechnen
    with st.status("Berechne Audiokomplexität...", expanded=True) as status:
        start_time = time.time()
        audio_complexity_results = extract_complexity(audio_file, original_tokens, device=str(device), batch_size=16) # Device-String
        df = pd.DataFrame(audio_complexity_results)
        df['position'] = [token['position'] for token in original_tokens]
        df['sentence'] = [token['sentence'] for token in original_tokens]
        end_time = time.time()
        runtime = end_time - start_time
        status.update(label=f"Audiokomplexität erfolgreich berechnet in {runtime:.2f} Sekunden", state="complete", expanded=False)

    with st.status("Isoliere Stimmen...", expanded=True) as status:
        start_time = time.time()
        isolate_speech(audio_file, device=str(device))
        aligned_results = improve_timesteps(f"{name}_vocals.wav", df, device=str(device), batch_size=16)
        df['start_time'] = [result['start_time'] for result in aligned_results]
        df['end_time'] = [result['end_time'] for result in aligned_results]
        end_time = time.time()
        runtime = end_time - start_time
        status.update(label=f"Stimmen erfolgreich isoliert in {runtime:.2f} Sekunden", state="complete", expanded=False)

    # 3. Wörter filtern und markieren
    with st.status("Filtere und markiere Wörter...", expanded=True) as status:
        start_time = time.time()
        df['display'] = None
        df['set_manually'] = False
        df['process'] = True
        exception_words = ["i", "no", "so"]
        df = mark_non_english_in_df(df, exception_words)
        df = mark_notes_in_df(df)
        df = mark_excluded_words(df)
        df = mark_numbers_in_df(df)
        end_time = time.time()
        runtime = end_time - start_time
        status.update(label=f"Wörter erfolgreich gefiltert und markiert in {runtime:.2f} Sekunden", state="complete", expanded=False)

    # 4. Übersetzung und Alignment
    with st.status("Übersetze und aligniere Sätze...", expanded=True) as status:
        start_time = time.time()
        translation_results = batch_translate_and_align(df, device=str(device)) 
        df_translations = pd.DataFrame(translation_results)
        df['translation'] = df_translations['german_translation']
        end_time = time.time()
        runtime = end_time - start_time
        status.update(label=f"Sätze erfolgreich übersetzt und aligned in {runtime:.2f} Sekunden", state="complete", expanded=False)

    # Initialize the FeatureExtractor with the device
    feature_extractor = FeatureExtractor(device=str(device))

    # 5. Features extrahieren
    with st.status("Extrahiere Features...", expanded=True) as status:
        start_time = time.time()
        
        # Word occurrence
        status.update(label="Berechne Wortvorkommen...", state="running", expanded=True)
        df.loc[df['process'], 'word_occurrence'] = df.loc[df['process'], 'word'].apply(feature_extractor.get_word_occurrence)
        
        # Word complexity
        status.update(label="Berechne Wortkomplexität...", state="running", expanded=True)
        df.loc[df['process'], 'word_complexity'] = df.loc[df['process'], 'word'].apply(feature_extractor.get_word_complexity)
        
        # Sentence complexity
        status.update(label="Berechne Satzkomplexität...", state="running", expanded=True)
        df_sentence_complexity = feature_extractor.get_sentence_complexity(df)
        df["sentence_complexity"] = df_sentence_complexity["sentence_complexity"]

        # Word importance
        status.update(label="Berechne Wortwichtigkeit...", state="running", expanded=True)
        word_importances = feature_extractor.get_word_importance(df, batch_size=32)
        df["word_importance"] = word_importances

        # Speech speed
        status.update(label="Berechne Sprechgeschwindigkeit...", state="running", expanded=True)
        df = feature_extractor.get_sentence_speed(df)

        # Speech volume
        status.update(label="Berechne Sprechlautstärke...", state="running", expanded=True)
        df = feature_extractor.get_sentence_speech_volume(df, name)

        # Ambient volume
        status.update(label="Berechne Umgebungslautstärke...", state="running", expanded=True)
        df = feature_extractor.get_word_ambient_volume(df, name)

        # Word frequency
        status.update(label="Berechne Wortfrequenz...", state="running", expanded=True)
        df = feature_extractor.get_word_frequency(df)

        end_time = time.time()
        runtime = end_time - start_time
        status.update(label=f"Features erfolgreich extrahiert in {runtime:.2f} Sekunden", state="complete", expanded=False)
        
    if predict:
        # 6. Vorhersagen treffen
        with st.status("Treffe Vorhersagen...", expanded=True) as status:
            start_time = time.time()
            # Set feature importance based on passed parameters
            df['audio_level'] = audio_level
            df['language_level'] = language_level
            df.loc[~df['set_manually'], 'display'] = predict_with_bias(df.loc[~df['set_manually']], device, bias=bias)
            end_time = time.time()
            runtime = end_time - start_time
            status.update(label=f"Vorhersagen erfolgreich getroffen in {runtime:.2f} Sekunden", state="complete", expanded=False)
            df.to_csv(f'{name}.csv', index=False)
            
        # 7. SRT generieren
        with st.status("Generiere Untertitel...", expanded=True) as status:
            start_time = time.time()
            generate_subtitles(name, df, srt_lines_in_memory)
            end_time = time.time()
            runtime = end_time - start_time
            status.update(label=f"Untertitel erfolgreich generiert in {runtime:.2f} Sekunden", state="complete", expanded=False)

    torch.cuda.empty_cache()
    return df, srt_lines_in_memory
    
def generate_subtitles(name, df, srt_lines_in_memory, english_level=0.0):
    languages = ["en", "de", "en-de", "de-en"]
    srt_files = {}
    
    for lang in languages:
        file_name = f"{name}[Filtered]_{lang}.srt"
        create_srt_file(
            srt_lines=srt_lines_in_memory,
            df=df,
            new_srt_file=file_name,
            original_timesteps=False,
            languages=lang.split('-'),
            english_level=english_level
        )
        srt_files[lang] = file_name
    return srt_files

def convert_srt_to_vtt(srt_content):
    """Convert SRT content to WebVTT format"""
    vtt = "WEBVTT\n\n"
    vtt += re.sub(r'(\d{2}:\d{2}:\d{2}),(\d{3})', r'\1.\2', srt_content)
    return vtt

def play_video(name, video_file=None, uploaded_srt_files=None):
    """
    Spielt ein Video mit Untertiteln ab und ermöglicht (optional) Speaker Boost.
    Unterstützt lokale und hochgeladene Dateien.
    
    Args:
        name: Name des Videos (ohne Erweiterung)
        video_file: Optional - hochgeladene Videodatei (st.UploadedFile)
        uploaded_srt_files: Optional - Liste der hochgeladenen SRT-Dateien
    """
    st.markdown("""
    <style>
    /* Beta badge styling */
    .beta-badge {
        background-color: #FF4B4B;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 6px;
    }
    </style>
    """, unsafe_allow_html=True)
    srt_files = {}
    video_path = None
    temp_files = []  # Temporäre Dateien zum späteren Aufräumen

    # Behandle hochgeladene Dateien
    if video_file and uploaded_srt_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name
            temp_files.append(video_path)
        for srt_upload in uploaded_srt_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
                tmp_srt.write(srt_upload.read())
                srt_path = tmp_srt.name
                temp_files.append(srt_path)
                if srt_upload.name == f"{name}[Filtered]_de-en.srt":
                    lang = "Englisch-Deutsch"
                elif srt_upload.name == f"{name}[Filtered]_en-de.srt":
                    lang = "Deutsch-Englisch"
                elif srt_upload.name == f"{name}[Filtered]_en.srt":
                    lang = "Englisch"
                elif srt_upload.name == f"{name}[Filtered]_de.srt":
                    lang = "Deutsch"
                elif srt_upload.name == f"{name}.srt":
                    lang = "Original"
                else:
                    lang = os.path.splitext(srt_upload.name)[0]
                srt_files[lang] = srt_path

    # Behandle lokale Dateien
    else:
        # Video-Datei suchen
        video_formats = [".mp4", ".mpeg", ".wav", ".mp3"]
        for format in video_formats:
            temp_video_file = f"{name}{format}"
            if os.path.exists(temp_video_file):
                video_path = temp_video_file
                break
        # Lokale SRT-Dateien suchen
        subtitle_files = {
            "Englisch": f"{name}[Filtered]_en.srt",
            "Deutsch": f"{name}[Filtered]_de.srt",
            "Englisch-Deutsch": f"{name}[Filtered]_en-de.srt",
            "Deutsch-Englisch": f"{name}[Filtered]_de-en.srt",
            "Original": f"{name}.srt",
        }
        for display_name, filename in subtitle_files.items():
            if os.path.isfile(filename):
                srt_files[display_name] = filename

    if not video_path:
        st.error("Keine Videodatei gefunden.")
        return

    # Konvertiere SRT-Dateien in VTT und erstelle track tags
    subtitle_tracks = ""
    subtitle_js = ""
    for idx, (lang, file_path) in enumerate(srt_files.items()):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            vtt_content = convert_srt_to_vtt(srt_content)
            vtt_b64 = base64.b64encode(vtt_content.encode('utf-8')).decode('utf-8')
            default_attr = ' default' if idx == 0 else ''
            subtitle_tracks += f'<track kind="subtitles" label="{lang}" src="data:text/vtt;base64,{vtt_b64}"{default_attr}>\n'
            if idx == 0:
                subtitle_js += f"""
                video.addEventListener('loadedmetadata', function() {{
                    if (video.textTracks.length > {idx}) {{
                        setTimeout(function() {{
                            video.textTracks[{idx}].mode = 'showing';
                        }}, 100);
                    }}
                }});
                """
        except Exception as e:
            st.error(f"Error loading subtitle {lang}: {str(e)}")

    # Lese und kodieren des Videos zu Base64
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
        return

    # Spalte für Speaker Boost (Beta)
    col1, col2 = st.columns(2)
    with col1:
            col3, col4 = st.columns([1, 3])
            with col3:
                st.markdown('<div style="display: flex; align-items: center; margin-top: 8px;">Speaker Boost <span class="beta-badge">BETA</span></div>', unsafe_allow_html=True)
            with col4:
                speaker_boost = st.toggle("", help="Experimental feature to enhance vocals in the video")
        
    if speaker_boost:
        with col2:
            slider_value = st.number_input(label="Vocals Volume", min_value=0.0, max_value=4.0,
                                       value=1.0, step=0.1, help="Relative Lautstärke der Stimmen gegenüber der Hintergrundlautstärke")
    else:
        slider_value = None

    # Falls Speaker Boost aktiviert ist, verarbeite die Audio-Dateien
    if speaker_boost:
        vocals_path = f"{name}_vocals.wav"
        no_vocals_path = f"{name}_no_vocals.wav"
        try:
            vocals = AudioSegment.from_wav(vocals_path)
            no_vocals = AudioSegment.from_wav(no_vocals_path)
            vocals_gain = slider_value * 10 if slider_value and slider_value >= 0 else 0
            bg_gain = 0 if slider_value and slider_value >= 0 else abs(slider_value) * 10
            adjusted_vocals = vocals.apply_gain(vocals_gain)
            adjusted_bg = no_vocals.apply_gain(bg_gain)
            min_duration = min(len(adjusted_vocals), len(adjusted_bg))
            adjusted_vocals = adjusted_vocals[:min_duration]
            adjusted_bg = adjusted_bg[:min_duration]
            mixed_audio = adjusted_bg.overlay(adjusted_vocals)
            audio_buffer = io.BytesIO()
            mixed_audio.export(audio_buffer, format="wav")
            audio_bytes = audio_buffer.getvalue()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            audio_html = f"""
            <audio id="myAudio">
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            """
            video_muted = "muted"
            audio_sync_js = f"""
            const video = document.getElementById('myVideo');
            const audio = document.getElementById('myAudio');
            {subtitle_js}
            video.addEventListener('play', function() {{
                audio.currentTime = video.currentTime;
                audio.play();
            }});
            video.addEventListener('pause', function() {{
                audio.pause();
            }});
            video.addEventListener('seeked', function() {{
                audio.currentTime = video.currentTime;
            }});
            setInterval(function() {{
                if (!video.paused && Math.abs(video.currentTime - audio.currentTime) > 0.3) {{
                    audio.currentTime = video.currentTime;
                }}
            }}, 500);
            """
        except Exception as e:
            st.error(f"Error processing audio for Speaker Boost: {str(e)}")
            video_muted = ""
            audio_html = ""
            audio_sync_js = subtitle_js
    else:
        video_muted = ""
        audio_html = ""
        audio_sync_js = subtitle_js

    # Erzeuge HTML-Code für Video, Untertitel und (optional) Audio
    html_code = f"""
    <html>
    <head>
        <style>
        video {{
            width: 100%;
        }}
        audio {{
            display: none;
        }}
        ::cue {{
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            font-size: 1.2em;
        }}
        </style>
    </head>
    <body>
        <video id="myVideo" controls {video_muted}>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            {subtitle_tracks}
            Your browser does not support the video tag.
        </video>
        {audio_html}
        <script>
        {audio_sync_js}
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=2000)

    # Aufräumen temporärer Dateien
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass