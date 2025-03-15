import string
import torch
import pandas as pd
from transformers import pipeline
from simalign import SentenceAligner
from collections import defaultdict
from datasets import Dataset  

def batch_translate_and_align(df: pd.DataFrame, device, batch_size=32):
    print(f"Using device: {device} for translation and alignment")
    translator = TranslationAligner(device=device)
    SPECIAL_TOKENS = {"♪": "♪"}
    
    # Reset index to use as original_index
    df = df.reset_index(drop=True)
    df['original_index'] = df.index

    # Build a mapping of sentence to corresponding tokens from the DataFrame.
    from collections import defaultdict
    sentence_to_tokens = defaultdict(list)
    for _, row in df.iterrows():
        token_data = {
            "token": row["word"],
            "original_sentence": row["sentence"],
            "position": row["position"],
            "original_index": row["original_index"]
        }
        sentence_to_tokens[row["sentence"]].append(token_data)
    
    unique_sentences = list(sentence_to_tokens.keys())

    # Process unique sentences using datasets for batching
    ds = Dataset.from_dict({"text": unique_sentences})
    translations_batch = translator.translator_pipeline(ds["text"], batch_size=batch_size)
    sentence_to_de = {sentence: trans["translation_text"] for sentence, trans in zip(unique_sentences, translations_batch)}

    alignment_info = {}
    for sentence_en in unique_sentences:
        token_list = sorted(sentence_to_tokens[sentence_en], key=lambda x: x["position"])
        src_tokens = [td["token"] for td in token_list]
        german_sent = sentence_to_de[sentence_en]
        tgt_tokens = german_sent.split()
        aligns = translator.aligner.get_word_aligns(src_tokens, tgt_tokens)
        alignment_pairs = aligns["inter"]
        alignment_info[sentence_en] = {
            "german_sentence": german_sent,
            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
            "alignment_pairs": alignment_pairs
        }

    # Collect all missing tokens across sentences first for fallback translation
    all_missing_tokens = set()
    for sentence_en, token_data_list in sentence_to_tokens.items():
        info = alignment_info[sentence_en]
        alignment_pairs = info["alignment_pairs"]
        token_data_list_sorted = sorted(token_data_list, key=lambda x: x["position"])
        for idx, token_data in enumerate(token_data_list_sorted):
            token = token_data["token"]
            if token in SPECIAL_TOKENS:
                continue
            aligned_indices = [tgt_idx for (src_idx, tgt_idx) in alignment_pairs if src_idx == idx]
            if not aligned_indices:
                all_missing_tokens.add(token)

    fallback_cache = {}
    if all_missing_tokens:
        missing_tokens_list = list(all_missing_tokens)
        ds_fallback = Dataset.from_dict({"text": missing_tokens_list})
        fallback_results = translator.translator_pipeline(ds_fallback["text"], batch_size=batch_size)
        for token, res in zip(missing_tokens_list, fallback_results):
            fallback_cache[token] = res["translation_text"]

    results = []
    # Build results using either the aligned or fallback translation.
    for sentence_en, token_data_list in sentence_to_tokens.items():
        info = alignment_info[sentence_en]
        german_sent = info["german_sentence"]
        tgt_tokens = info["tgt_tokens"]
        alignment_pairs = info["alignment_pairs"]
        token_data_list_sorted = sorted(token_data_list, key=lambda x: x["position"])
        for idx, token_data in enumerate(token_data_list_sorted):
            token = token_data["token"]
            if token in SPECIAL_TOKENS:
                aligned_words_cleaned = SPECIAL_TOKENS[token]
            else:
                aligned_indices = [tgt_idx for (src_idx, tgt_idx) in alignment_pairs if src_idx == idx]
                if not aligned_indices:
                    aligned_words_cleaned = fallback_cache.get(token, token)
                else:
                    aligned_words = [
                        tgt_tokens[t_i]
                        for t_i in aligned_indices
                        if 0 <= t_i < len(tgt_tokens)
                    ]
                    aligned_words_cleaned = " ".join(
                        w.translate(str.maketrans("", "", string.punctuation))
                        for w in aligned_words
                    )
            results.append({
                "original_index": token_data["original_index"],
                "english_token": token,
                "english_sentence": sentence_en,
                "german_translation": aligned_words_cleaned,
                "german_full_sentence": german_sent
            })

    results_sorted = sorted(results, key=lambda x: x["original_index"])
    return results_sorted

class TranslationAligner:
    def __init__(self, model="bert", token_type="bpe", matching_methods="mai", device=None):
        self.device = device
        self.translator_pipeline = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-de",
            device=self.device,
            max_length=512,
            early_stopping=True
        )
        self.aligner = SentenceAligner(
            model=model,
            token_type=token_type,
            matching_methods=matching_methods,
            device=self.device
        )