import pandas as pd
from fast_langdetect import detect
from utils import EXCLUDED_WORDS

def detect_tokens_non_english(original_tokens, exception_words):
    # Schritt (1): Sprache nur 1x pro Satz bestimmen (Cache)
    unique_sentences = {item['original_sentence'] for item in original_tokens}
    sentence_language_cache = {}
    for sentence in unique_sentences:
        cleaned_sentence = sentence.replace("\n", "").strip()
        if cleaned_sentence:
            result = detect(cleaned_sentence, low_memory=False)
            sentence_language_cache[sentence] = result["lang"]
        else:
            sentence_language_cache[sentence] = None  # Keine Sprache ermittelbar

    # Schritt (2): Für jedes Token schauen, ob Satz != 'en'/'la';
    #               falls ja, Token einzeln prüfen, sonst direkt False.
    token_info = []
    for item in original_tokens:
        sent_lang = sentence_language_cache.get(item["original_sentence"], None)

        # Falls Satz als English oder Latein erkannt -> alles auf False
        if sent_lang in ("en", "la"):
            token_info.append({
                "word": item["token"],
                "display": False,
                "set_manually": False
            })
        else:
            # Satzsprache ist nicht en/la -> Token einzeln prüfen
            token_text = item["token"].strip()
            if token_text:
                if token_text in exception_words:
                    # Wenn das Wort in der Ausnahmeliste ist, dann ebenfalls alles False
                    token_info.append({
                        "word": item["token"],
                        "display": False,
                        "set_manually": False
                    })
                else:
                    word_detection = detect(token_text, low_memory=False)
                    word_lang = word_detection["lang"]
                    if word_lang in ("en", "la"):
                        # Wenn das Wort Englisch/Latein ist, dann ebenfalls alles False
                        token_info.append({
                            "word": item["token"],
                            "display": False,
                            "set_manually": False
                        })
                    else:
                        # Wort ist nicht Englisch oder Latein
                        token_info.append({
                            "word": item["token"],
                            "display": True,
                            "set_manually": True
                        })
                        print(f"Token '{item['token']}' in Satz '{item['original_sentence']}' ist nicht Englisch/Latein.")
            else:
                # Leeres Token -> optional alles False
                token_info.append({
                    "word": item["token"],
                    "display": False,
                    "set_manually": False
                })

    return pd.DataFrame(token_info)

def mark_non_english_in_df(df: pd.DataFrame, exception_words=["i", "no"]) -> pd.DataFrame:
    # Build original_tokens from df rows using the "word" and "sentence" columns.
    original_tokens = df.apply(lambda row: {"token": row["word"], "original_sentence": row["sentence"]}, axis=1).tolist()
    df_language = detect_tokens_non_english(original_tokens, exception_words)
    df["display"] = df_language["display"]
    df["set_manually"] = df_language["set_manually"]
    return df

def mark_notes_in_df(df):
    skip = False
    marked_words = []

    for idx, row in df.iterrows():
        if row["word"] == "♪":
            df.at[idx, 'display'] = True
            df.at[idx, 'set_manually'] = True
            df.at[idx, 'process'] = False
            marked_words.append(row["word"])
            skip = not skip
            continue

        if skip:
            df.at[idx, 'display'] = True
            df.at[idx, 'set_manually'] = True
            df.at[idx, 'process'] = False
            marked_words.append(row["word"])

    if marked_words:
        print("\nIncluded words:")
        print(", ".join(marked_words))
    return df

def mark_excluded_words(df):
    total_excluded = 0
    mask = df['word'].isin(EXCLUDED_WORDS)
    df.loc[mask, ['display', 'set_manually', 'process']] = [False, True, True]
    total_excluded = mask.sum()
    print(f"\nExcluded {total_excluded} words based on EXCLUDED_WORDS list")
    return df

def mark_numbers_in_df(df):
    marked_words = []
    for idx, row in df.iterrows():
        try:
            num = float(row['word'])
            if num > 13:
                df.at[idx, 'display'] = True
                df.at[idx, 'set_manually'] = True
                df.at[idx, 'process'] = True
                marked_words.append(row['word'])
        except ValueError:
            continue
    if marked_words:
        print("\nIncluded numbers:")
        print(", ".join(marked_words))
    return df