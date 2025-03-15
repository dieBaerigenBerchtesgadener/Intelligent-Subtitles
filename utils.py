import string
import torch
import re
from typing import Optional

# Konstanten
AUDIO_TYPE = ".mp4"
EXCLUDED_WORDS = [
    "mm", "nah", "uh", "um", "whoa", "uhm", "uhmm", "ah", "mm-hmm",
    "uh-huh", "uh-uh", "uh-uhm", "uh-uhmm", "uhm-h", "aw", "ugh",
    "shh", "mmhmm", "huh", "hmm", "mmm", "oops", "oopsie", "uh-oh",
    "whoops", "oof", "yup", "yep", "nope", "aha", "tsk", "ew", "phew",
    "meh", "huh-uh", "huh-huh", "huh-uhm", "mhm", "oh", "hmm-m",
    "er", "eh", "ahh", "yikes", "yawn", "ugh-ugh", "jeez", "duh",
    "wow", "meh-meh", "uhhh", "ummm", "ugh-huh", "hmpf", "yawn-yawn",
    "heh", "hmph", "eep", "gah", "uhp", "boo", "psst", "argh", "oi",
    "ohh", "oh-ho", "whoa-whoa", "la", "laa", "ah-ha", "ha", "ha-ha",
    "hahaha", "bah", "whew", "ehh", "huff", "uff", "sniff", "snort",
    "gulp", "hic", "haah", "bleh", "blah", "bla", "mwaa", "uhuh",
    "yah", "uhw", "eww", "ewww", "grr", "huh-huh", "haha", "shush",
    "wha", "wham", "bam", "oooh", "aaah", "hrr", "uhhhhhh", "ummmmm",
    "woah", "ughugh", "mm-mm", "uh-huh-huh", "erm", "grrr", "urr",
    "yippie", "oops-a-daisy", "ouch", "eek", "zoinks", "woopsie",
    "yeesh", "hm-mm", "uhhuh", "hrrmph", "bleugh", "rawr", "ick",
    "whaa", "la-la", "meep", "pfft", "haaa", "ahhhhhh", "oii", "tsk-tsk",
    "blub", "blurgh", "brr", "rrr", "oomph", "ohhhhhh", "hmmmmmm",
    "ahhhhhhh", "guh", "ack", "zzzz", "hush", "hsh", "boo-hoo", "ho-hum",
    "urrgh", "grumble", "murmur", "mutter", "uhhhhmm", "hah", "ah-ah",
    "shoo", "la-la-la", "blah-blah", "tra-la", "lalala", "waah", "waaah",
    "ooh-ooh", "uhh", "uhhhh", "erhm", "ermm", "urrggh", "aargh",
    "hm-mm-mm", "uh-uh-uh", "uhm-uh", "hurmph", "grmph", "ha-umph",
    "um-hum", "humph", "shhhhhh", "psssh", "whisper", "moan", "groan",
    "ah-choo", "cough", "sneeze", "hick", "hiccup", "snore", "whaaat",
    "doh", "hmh", "pfft-pfft", "chatter", "rumble", "buzz", "mumble",
    "ooh-la-la", "ahem", "tut", "hrrmm", "grmph", "sigh", "gulp-gulp",
    "oh-wow", "yeehaw", "oh-no", "ach", "achoo", "whoop", "zipp", "zzz", "so"
]
# CEFR levels mapping
cefr_levels = {
    'A1': 0.0,
    'A2': 0.2,
    'B1': 0.4,
    'B2': 0.6,
    'C1': 0.8,
    'C2': 1.0
}

device = torch.device("cuda")
print(f"Using device: {device}")

def clean_token(token: str) -> Optional[str]:
    """
    Bereinigt einzelne Wörter:
    - Führende/abschließende Satzzeichen entfernen
    - Kleinbuchstaben
    - Leerstrings -> None
    """
    token = token.strip(string.punctuation + '""\'').lower()
    return token if token else None

def remove_punctuation_not_between_letters(text: str) -> str:
    """
    Entfernt Satzzeichen (? , ; . -) nur dann, wenn sie nicht
    zwischen zwei alphanumerischen Zeichen stehen [1].
    """
    return re.sub(r'(?<![A-Za-z0-9])[?;.,=-]+|[?;.,=-]+(?![A-Za-z0-9])', '', text)

def seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Konvertiert eine Zeit in Sekunden in das SRT-Zeitformat "HH:MM:SS,mmm" [1].
    """
    if seconds is None or seconds != seconds: # Check for NaN
        return None
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def clean_word(token):
    return token.strip(string.punctuation + '“”’').replace(" ", "").replace("’", "'").lower()

# Hilfsfunktion: Entfernt Apostrophe für Vergleich
def remove_apostrophes(word):
    return re.sub(r"[’']", "", word).lower()
