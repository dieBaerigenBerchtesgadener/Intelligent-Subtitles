import string
from faster_whisper import WhisperModel, BatchedInferencePipeline
import difflib
import contractions
from utils import clean_word, remove_apostrophes
import threading

# Globaler Modell-Cache und Lock für thread-sicheren Zugriff
_GLOBAL_MODEL = None
_MODEL_LOCK = threading.Lock()

def get_model(device="cpu", compute_type=None):
    """
    Liefert ein singleton WhisperModel. Falls eine CUDA-GPU genutzt wird,
    wird als compute_type "float16" verwendet, sofern nicht explizit anders angegeben.
    """
    global _GLOBAL_MODEL
    with _MODEL_LOCK:
        if _GLOBAL_MODEL is None:
            if device != "cpu":
                compute_type = compute_type or "int8"
                # Bei GPU können Threads nicht stets voll ausgenutzt werden
                cpu_threads = 1
                num_workers = 6
            else:
                compute_type = compute_type or "int8"
                cpu_threads = 1
                num_workers = 1

            print(f"Initializing WhisperModel on {device} with compute_type {compute_type}")
            _GLOBAL_MODEL = WhisperModel(
                "tiny",
                device=device,
                compute_type=compute_type
            )
        return _GLOBAL_MODEL

def extract_complexity(audio_file, original_tokens, device="cpu", batch_size=16, compute_type=None):
    """
    Extrahiert die Audio-Komplexität basierend auf der Differenz zwischen den Original- und den 
    mittels WhisperModel erkannten Tokens. Nutzt ein global initialisiertes Modell zur 
    Vermeidung von wiederholten Initialisierungen.
    """
    print(f"Using device: {device} for audio complexity extraction")
    model = get_model(device=device, compute_type=compute_type)
    batched_model = BatchedInferencePipeline(model=model)
    print("Getting transcription from audio file...")
    segments, info = batched_model.transcribe(
        audio_file,
        word_timestamps=True,
        beam_size=1,
        batch_size=batch_size
    )

    # Extrahiere Tokens mit Wahrscheinlichkeit und Zeitstempel
    predicted_tokens = []
    for segment in segments:
        for word_info in segment.words:
            cleaned = clean_word(word_info.word)
            probability = word_info.probability
            predicted_tokens.append({
                'token': cleaned,
                'probability': probability
            })

    original_sequence = [t['token'] for t in original_tokens]
    predicted_sequence = [t['token'] for t in predicted_tokens]

    # Sequenz-Ausrichtung mittels difflib.SequenceMatcher
    matcher = difflib.SequenceMatcher(None, original_sequence, predicted_sequence)
    aligned_results = []

    for opcode in matcher.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag == 'equal':
            for idx_orig, idx_pred in zip(range(i1, i2), range(j1, j2)):
                aligned_results.append({
                    'word': original_tokens[idx_orig]['token'],
                    'audio_complexity': 1 - predicted_tokens[idx_pred]['probability']
                })

        elif tag == 'replace':
            orig_joined = " ".join(original_sequence[i1:i2]).lower()
            pred_joined = " ".join(predicted_sequence[j1:j2]).lower()

            # Kontraktion behandeln (z. B. "he's" ↔ "he is")
            if len(predicted_sequence[j1:j2]) == 1 and contractions.fix(predicted_sequence[j1]) == orig_joined:
                pred = predicted_tokens[j1]
                for idx_orig in range(i1, i2):
                    aligned_results.append({
                        'word': original_tokens[idx_orig]['token'],
                        'audio_complexity': 1 - pred['probability']
                    })

            # Vergleich ohne Apostrophe (z. B. "name's" ↔ "names")
            elif remove_apostrophes(orig_joined) == remove_apostrophes(pred_joined):
                pred = predicted_tokens[j1]
                for idx_orig in range(i1, i2):
                    aligned_results.append({
                        'word': original_tokens[idx_orig]['token'],
                        'audio_complexity': 1 - pred['probability']
                    })

            else:
                for idx_orig in range(i1, i2):
                    aligned_results.append({
                        'word': original_tokens[idx_orig]['token'],
                        'audio_complexity': 1.0
                    })

        elif tag == 'delete':
            for idx_orig in range(i1, i2):
                aligned_results.append({
                    'word': original_tokens[idx_orig]['token'],
                    'audio_complexity': 1.0
                })

        elif tag == 'insert':
            # Überspringe zusätzliche Tokens in der Vorhersage
            pass

    return aligned_results

def improve_timesteps(audio_file, df, device="cpu", batch_size=16):
    """
    Nutzt ein verbessertes Modell (base.en) um genauere Zeitstempel (start_time und end_time)
    für jedes Wort aus dem Audio zu extrahieren. Die Wahrscheinlichkeiten werden dabei nicht verändert.
    
    Args:
        audio_file (str): Pfad zur Audiodatei.
        df (pd.DataFrame): DataFrame, der mindestens die Spalte "word" enthält.
        device (str): Gerät für die Verarbeitung, z.B. "cpu" oder "cuda".
        batch_size (int): Batch-Größe für die Verarbeitung.
    
    Returns:
        list: Liste von Dictionaries, in denen jedes Wort den Originaltoken, die Audio-Komplexität
              (1 - probability), sowie start_time und end_time enthält.
    """
    print(f"Using device: {device} for improved timestep extraction")
    # Erzeuge das verbesserte Modell
    model = WhisperModel("distil-medium.en", device=device, compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)

    print("Getting improved transcription from audio file...")
    segments, info = batched_model.transcribe(
        audio_file,
        word_timestamps=True,
        batch_size=batch_size
    )

    predicted_tokens = []
    for segment in segments:
        for word_info in segment.words:
            cleaned = clean_word(word_info.word)
            start_time = word_info.start
            end_time = word_info.end
            predicted_tokens.append({
                'token': cleaned,
                'start': start_time,
                'end': end_time,
            })

    original_sequence = df['word'].tolist()
    predicted_sequence = [t['token'] for t in predicted_tokens]

    matcher = difflib.SequenceMatcher(None, original_sequence, predicted_sequence)
    aligned_results = []

    for opcode in matcher.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag == 'equal':
            for idx_orig, idx_pred in zip(range(i1, i2), range(j1, j2)):
                aligned_results.append({
                    'word': original_sequence[idx_orig],
                    'start_time': predicted_tokens[idx_pred]['start'],
                    'end_time': predicted_tokens[idx_pred]['end']
                })
        elif tag == 'replace':
            orig_joined = " ".join(original_sequence[i1:i2]).lower()
            pred_joined = " ".join(predicted_sequence[j1:j2]).lower()

            # Kontraktion behandeln (z. B. "he's" ↔ "he is")
            if len(predicted_sequence[j1:j2]) == 1 and contractions.fix(predicted_sequence[j1]) == orig_joined:
                pred = predicted_tokens[j1]
                for idx_orig in range(i1, i2):
                    aligned_results.append({
                        'word': original_sequence[idx_orig],
                        'start_time': pred['start'],
                        'end_time': pred['end']
                    })
            # Vergleich ohne Apostrophe (z. B. "name's" ↔ "names")
            elif remove_apostrophes(orig_joined) == remove_apostrophes(pred_joined):
                pred = predicted_tokens[j1]
                for idx_orig in range(i1, i2):
                    aligned_results.append({
                        'word': original_sequence[idx_orig],
                        'start_time': pred['start'],
                        'end_time': pred['end']
                    })
            else:
                for idx_orig in range(i1, i2):
                    aligned_results.append({
                        'word': original_sequence[idx_orig],
                        'start_time': None,
                        'end_time': None
                    })
        elif tag == 'delete':
            for idx_orig in range(i1, i2):
                aligned_results.append({
                    'word': original_sequence[idx_orig],
                    'start_time': None,
                    'end_time': None
                })
        elif tag == 'insert':
            # Zusätzliche Tokens aus der Vorhersage überspringen
            pass

    return aligned_results