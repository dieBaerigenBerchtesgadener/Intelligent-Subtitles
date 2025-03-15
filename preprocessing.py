import re
import string
from typing import Optional
from nltk.tokenize import sent_tokenize
import nltk
from utils import clean_token
import os
import numpy as np
import torch
import torchaudio
import tempfile
import subprocess
from scipy.io import wavfile
from demucs.pretrained import get_model
from demucs.apply import apply_model
#nltk.download('punkt', quiet=True)
from demucs import separate

def create_bracketless_lines(original_srt_file: str) -> list[str]:
    """
    Liest das Original-SRT zeilenweise ein und entfernt nur in Textzeilen
    sämtliche Inhalte in Klammern (...), [...], {...}.
    Zeitstempel, Zeilennummern, Leerzeilen usw. bleiben unverändert.
    Gibt die so modifizierten Zeilen als Liste zurück.
    """
    bracketless_lines = []
    with open(original_srt_file, "r", encoding="utf-8") as f:
        for line in f:
            # Nur Textzeilen anpassen (nicht Zeitstempel, Nummern oder Leerzeilen)
            if '-->' not in line and not line.strip().isdigit() and line.strip():
                line = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', line)
            bracketless_lines.append(line)
    return bracketless_lines

def remove_brackets_in_text(line: str) -> str:
    """
    Entfernt Text in (), [] und {} aus einer Textzeile.
    """
    return re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', line)

def read_srt_in_memory(srt_path: str) -> list[str]:
    """
    Liest die Original-SRT-Datei in eine Liste von Zeilen ein.
    Entfernt Klammerinhalte NUR in Zeilen, die kein Zeitstempel
    und keine reine Nummernzeilen sind. Leerzeilen oder Zeitzeilen
    bleiben unverändert.
    """
    lines = []
    with open(srt_path, "r", encoding="utf-8") as f:
        for line in f:
            if '-->' not in line and not line.strip().isdigit() and line.strip():
                # Das ist eine "Text"-Zeile -> Klammern entfernen
                line = remove_brackets_in_text(line)
            lines.append(line)
    return lines

def custom_split_into_sentences(text: str) -> list[str]:
    """
    Sucht zuerst Paare von Notenzeichen (♪), sodass alles zwischen
    zwei ♪ (inklusive) zu EINEM Satzsegment wird, z.B. "♪ Yeah, baby ♪".
    Alles, was nicht in ein solches Paar fällt, wird normal weiter
    verarbeitet. Danach wird jedes Segment via sent_tokenize weiter
    zerteilt, falls noch zusätzliche Satzgrenzen existieren.
    """
    # Zuerst splitten wir nach ♪, behalten sie aber in der Liste (Capture Group).
    raw_parts = re.split(r'(♪)', text)

    merged_segments = []
    i = 0
    while i < len(raw_parts):
        current = raw_parts[i].strip()

        # Prüfen, ob dieses Element ein Notenzeichen ist und ob
        # in raw_parts[i+2] wieder ein ♪ vorhanden ist
        if current == '♪' and (i + 2) < len(raw_parts) and raw_parts[i+2].strip() == '♪':
            # Beispiel: ["♪", " Yeah, baby ", "♪"]
            inner_text = raw_parts[i+1].strip() if (i+1 < len(raw_parts)) else ""
            merged_segments.append(f'♪ {inner_text} ♪')
            i += 3  # Überspringen wir die drei genutzten Elemente
        else:
            # Falls kein ♪-Paar vorliegt, nur das aktuelle Element übernehmen (ggf. Leerstrings ignorieren)
            if current:
                merged_segments.append(current)
            i += 1

    # Jetzt liegt in merged_segments entweder "♪ Yeah, baby ♪"
    # oder "irgendwelcher Text ohne Notenzeichen" oder beides gemischt.
    # Danach zerteilen wir jedes Segment vorsichtshalber noch mittels sent_tokenize
    # (falls z.B. in "♪ ..." ein Punkt oder Fragezeichen vorkommt).
    final_sentences = []
    for segment in merged_segments:
        # segment könnte z.B. "♪ Yeah, baby ♪" ODER "Das ist ein Satz. Und noch einer." sein
        for s in sent_tokenize(segment):
            s = s.strip()
            if s:
                final_sentences.append(s)

    return final_sentences

def extract_tokens_with_sentences(srt_lines: list[str]) -> list[dict]:
    """
    Extrahiert Tokens und zugehörige originale Sätze aus den SRT-Zeilen,
    wobei Sätze, die sich über mehrere Zeilen erstrecken, korrekt
    behandelt werden. Zusätzlich wird der Text bei Notenzeichen-Paaren
    in einen eigenen Satz "♪ ... ♪" eingefasst.
    """
    tokens_with_sentences = []
    full_text = ""

    # Kombiniere alle Textzeilen zu einem einzigen String
    for line in srt_lines:
        if '-->' not in line and not line.strip().isdigit() and line.strip():
            cleaned_line = line.strip().lstrip('-').strip()
            full_text += cleaned_line + " "

    # Segmentiere an Notenzeichen-Paaren und Satzzeichen
    sentences = custom_split_into_sentences(full_text)

    # Extrahiere Tokens aus jedem Satz und speichere den Originalsatz
    for sentence in sentences:
        words = sentence.split()
        position = 0
        for w in words:
            position += 1
            t = clean_token(w)
            if t:
                tokens_with_sentences.append({
                    'token': t,
                    'sentence': sentence,
                    'position': position
                })

    return tokens_with_sentences

def isolate_speechDEPRECATED(audio_file, device, output_dir="data"):
    """
    Process audio with Demucs to separate vocals from other audio components.
    
    Args:
        audio_file: Path to the input audio or video file
        output_dir: Directory to save the processed audio files (default: "data")
    
    Returns:
        Tuple of paths to the processed files (vocals_path, no_vocals_path)
    """
    
    # Load model
    model = get_model("htdemucs").to(device)
    print("Using device to process audio:", device)
    sample_rate = model.samplerate
    
    def extract_audio_from_video(file_path):
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        cmd = ['ffmpeg', '-i', file_path, '-vn', '-acodec', 'pcm_s16le', 
               '-ar', str(sample_rate), '-ac', '2', '-y', temp_wav.name]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            waveform, sr = torchaudio.load(temp_wav.name)
            return waveform, sample_rate
        except Exception:
            return None, None
        finally:
            try:
                os.unlink(temp_wav.name)
            except:
                pass
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract file name without extension
    name = os.path.splitext(os.path.basename(audio_file))[0]
    
    # Extract audio from video/audio file
    waveform, sr = extract_audio_from_video(audio_file)
    if waveform is None:
        return None, None
        
    # Ensure correct format
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    
    # Process audio
    waveform = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        sources = apply_model(model, waveform)
    
    # Extract and save vocals
    vocal_idx = model.sources.index("vocals")
    vocals = sources[0, vocal_idx].cpu()
    vocals_np = (vocals * 32767).numpy().astype(np.int16)
    vocals_path = os.path.join(output_dir, name + "_vocals.wav")
    wavfile.write(vocals_path, sample_rate, vocals_np.T)
    
    # Create and save no_vocals track
    no_vocals = torch.zeros_like(sources[0, 0])
    for i in range(len(model.sources)):
        if i != vocal_idx:
            no_vocals += sources[0, i]
    no_vocals_np = (no_vocals.cpu() * 32767).numpy().astype(np.int16)
    no_vocals_path = os.path.join(output_dir, name + "_no_vocals.wav")
    wavfile.write(no_vocals_path, sample_rate, no_vocals_np.T)
    print("Vocals and no-vocals tracks saved to:", vocals_path, no_vocals_path)
    return vocals_path, no_vocals_path


def isolate_speech(input_file, device, output_dir="data"):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    args = [
        "--two-stems=vocals",
        "--device", device,
        "--out", output_dir,
        "--filename", f"{base_name}_{{stem}}.wav",
        input_file
    ]
    
    print("Running Demucs with arguments:", args)
    
    separate.main(args)
    source_folder = os.path.join(output_dir, "htdemucs")
    vocals_src = os.path.join(source_folder, f"{base_name}_vocals.wav")
    no_vocals_src = os.path.join(source_folder, f"{base_name}_no_vocals.wav")

    vocals_dest = os.path.join(output_dir, f"{base_name}_vocals.wav")
    no_vocals_dest = os.path.join(output_dir, f"{base_name}_no_vocals.wav")

    try:
        os.replace(vocals_src, vocals_dest)
        os.replace(no_vocals_src, no_vocals_dest)
        os.rmdir(source_folder)  # Remove the now empty htdemucs folder
    except OSError as e:
        print(f"Error moving files or removing directory: {e}")
    
    print(f"Separation complete. Files saved in {output_dir}:")
    print(f"- {base_name}_vocals.wav")
    print(f"- {base_name}_no_vocals.wav")
