import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from scipy.stats import entropy
from utils import cefr_levels
from cefrpy import CEFRAnalyzer
import re
import numpy as np
import nltk
# Uncomment the following line if running for the first time
# nltk.download('cmudict')
from nltk.corpus import cmudict
import librosa
import os
from wordfreq import zipf_frequency

class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        print(f"Using {self.device} for feature extraction.")
        self.model_id = "answerdotai/ModernBERT-base"
        self.tokenizer_mlm = AutoTokenizer.from_pretrained(self.model_id)
        self.model_mlm = AutoModelForMaskedLM.from_pretrained(self.model_id).to(self.device)
        # Enable half precision on GPU for faster inference.
        if self.device != "cpu":
            self.model_mlm.half()
        
        # Internal caches
        self.word_complexity_cache = {}
        self.entropy_cache = {}
        self.word_counts = {}
        # Audio attributes to avoid loading vocals repeatedly.
        self.vocals_audio = None
        self.sr_vocals = None

    def load_vocals(self, name):
        """
        Loads vocals audio from file if not already loaded.
        """
        if self.vocals_audio is None or self.sr_vocals is None:
            vocals_path = os.path.join(f"{name}_vocals.wav")
            print("Loading vocals audio from:", vocals_path)
            self.vocals_audio, self.sr_vocals = librosa.load(vocals_path, sr=None)

    def get_word_complexity(self, word):
        if word in self.word_complexity_cache:
            return self.word_complexity_cache[word]
        analyzer = CEFRAnalyzer()
        try:
            level = analyzer.get_average_word_level_CEFR(word)
            if level is not None:
                score = cefr_levels.get(level.name, 0.0)
                self.word_complexity_cache[word] = score
                return score
        except Exception as e:
            print(f"Error retrieving level for word '{word}': {e}")
            self.word_complexity_cache[word] = 0.0
            return 0.0
        self.word_complexity_cache[word] = 0.0
        return 0.0

    def get_word_occurrence(self, word, max_count=15):
        self.word_counts[word] = self.word_counts.get(word, 0) + 1
        if self.word_counts[word] >= max_count:
            return 1.0
        return (self.word_counts[word] - 1) / (max_count - 1)

    
    def get_word_importance(self, df: pd.DataFrame, batch_size=32):
        """
        Optimized version that batches tokenization and forward passes using a DataFrame directly.
        It expects a DataFrame containing at least the columns:
        "word", "sentence", "position", and "process".
        The final computed importance value remains the same.
        """
        # Build original_tokens from the DataFrame rows.
        original_tokens = df.apply(
            lambda row: {
                "token": row["word"],
                "original_sentence": row["sentence"],
                "position": row["position"]
            },
            axis=1
        ).tolist()

        importances = []
        for i in range(0, len(original_tokens), batch_size):
            batch_tokens = original_tokens[i: i + batch_size]
            # Prepare a result list for the current batch.
            batch_results = [None] * len(batch_tokens)
            valid_indices = []
            valid_tokens = []
            
            # Collect only tokens that need processing.
            for j, token_data in enumerate(batch_tokens):
                if not df.loc[i + j, "process"]:
                    batch_results[j] = None
                else:
                    valid_indices.append(j)
                    valid_tokens.append(token_data)
            
            if valid_tokens:
                # Batch tokenize the sentences.
                sentences = [token_data["original_sentence"] for token_data in valid_tokens]
                encoding = self.tokenizer_mlm(
                    sentences,
                    return_tensors='pt',
                    add_special_tokens=True,
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                # Retrieve word_ids per sentence.
                word_ids_list = [
                    encoding.word_ids(batch_index=k) for k in range(len(sentences))
                ]
                # Save the original token IDs for later probability lookup.
                original_input_ids = encoding['input_ids'].clone()
                
                valid_subtoken_indices = []
                # Locate subtoken positions and replace them with the mask token.
                for idx, token_data in enumerate(valid_tokens):
                    target_word_id = token_data["position"] - 1
                    subtoken_indices = [pos for pos, w_id in enumerate(word_ids_list[idx])
                                        if w_id == target_word_id]
                    valid_subtoken_indices.append(subtoken_indices)
                    for pos in subtoken_indices:
                        encoding['input_ids'][idx, pos] = self.tokenizer_mlm.mask_token_id
                        
                # Move tensors to the target device.
                encoding = {k: v.to(self.device, non_blocking=True)
                            for k, v in encoding.items() if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    outputs = self.model_mlm(**encoding)
                logits = outputs.logits
                
                # Compute the importance for each valid token.
                for idx, subtoken_indices in enumerate(valid_subtoken_indices):
                    if not subtoken_indices:
                        importance = 0.0
                    else:
                        prob_product = 1.0
                        for pos in subtoken_indices:
                            softmax_probs = torch.softmax(logits[idx, pos], dim=-1)
                            original_token_id = original_input_ids[idx, pos]
                            prob_product *= softmax_probs[original_token_id].item()
                        importance = 1.0 - prob_product
                    # Insert the computed importance in its original position.
                    batch_results[valid_indices[idx]] = importance
            importances.extend(batch_results)
        return importances

    def calculate_entropy(self, text, tokenizer):
        """
        Calculate model-based entropy using MLM predictions.
        
        Args:
            text (str): Input text to calculate entropy for
            tokenizer: The tokenizer instance (not used in this implementation since we use self.tokenizer_mlm)
        
        Returns:
            float: Calculated entropy value
        """
        # Prepare input
        inputs = self.tokenizer_mlm(
            text,
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model_mlm(**inputs)
        
        # Calculate probabilities for each position
        predictions = torch.softmax(outputs.logits, dim=-1)
        
        # Get probabilities for actual tokens
        token_probs = []
        for i, token_id in enumerate(inputs.input_ids[0]):
            token_prob = predictions[0, i, token_id].item()
            token_probs.append(token_prob)
        return entropy(token_probs)
    
    def get_sentence_complexity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentence complexities using GPU acceleration where possible.
        
        Args:
            original_tokens: List of dictionaries containing token information
            
        Returns:
            pd.DataFrame: DataFrame containing word entries with complexity scores
        """
        # Use the already initialized tokenizer from the class
        tokenizer = self.tokenizer_mlm
        
        # Reference sentence for max entropy calculation
        extremely_complex_sentence = "I used to believe that technology could save us from the climate crisis, that all the big brains in the world would come up with a silver bullet to stop carbon pollution, that a clever policy would help that technology spread, and our concern about the greenhouse gases heating the planet would be a thing of the past, and we wouldn't have to worry about the polar bears anymore."
        
        # Calculate max entropy once
        max_entropy = self.calculate_entropy(extremely_complex_sentence, tokenizer)
        print(f"Max entropy: {max_entropy}")
        
        # Get unique sentences from the DataFrame
        unique_sentences = list(df["sentence"].unique())
        
        # Create cache for sentence complexities
        sentence_complexity_cache = {}
        batch_size = 32
        for i in range(0, len(unique_sentences), batch_size):
            batch_sentences = unique_sentences[i:i + batch_size]
            for sentence in batch_sentences:
                if sentence not in sentence_complexity_cache:
                    entropy_value = self.calculate_entropy(sentence, tokenizer)
                    normalized_entropy = entropy_value / max_entropy
                    sentence_complexity_cache[sentence] = normalized_entropy
        
        # Map each row in df to its sentence complexity and create the output DataFrame.
        df_sentence_complexity = df.copy()
        df_sentence_complexity["sentence_complexity"] = df_sentence_complexity["sentence"].map(sentence_complexity_cache)
        # Return a DataFrame with the word and its computed sentence complexity.
        return df_sentence_complexity[["word", "sentence_complexity"]]
    
        
    def get_sentence_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame containing words and timings to compute syllables-per-second
        metrics and a normalized speed value based on a sigmoid function.
        
        Assumes the DataFrame has the following columns:
        - "word": the text of the word.
        - "position": numerical order in the sentence (with each sentence starting with 1).
        - "start_time" and "end_time": timing values (may contain nan).
        
        Returns:
            The input DataFrame with a new 'speed' column.
        """

        cmu_dict = cmudict.dict()

        def count_syllables(word):
            """
            Count syllables using the CMU dictionary.
            If a word is not available, falls back to counting vowel-group matches.
            """
            word = word.lower().strip(".,;:'\"!?")
            if word in cmu_dict:
                pronunciation = cmu_dict[word][0]  # use first pronunciation
                return sum(1 for ph in pronunciation if ph[-1].isdigit())
            else:
                return len(re.findall(r'[aeiouy]+', word))

        def compute_sentence_metrics(group):
            """
            Compute duration and syllables-per-second for a sentence group.
            """
            group = group.sort_values(by='position').reset_index(drop=True)
            valid_start = group[group['start_time'].notna()]
            valid_end = group[group['end_time'].notna()]
            if not valid_start.empty and not valid_end.empty:
                # Use the first valid start_time and the last valid end_time.
                start_time = valid_start.iloc[0]['start_time']
                end_time = valid_end.iloc[-1]['end_time']
                duration = end_time - start_time
                # Compute syllable count for words between the first and last valid timing.
                start_idx = group.index[group['start_time'].notna()][0]
                end_idx = group.index[group['end_time'].notna()][-1]
                syllable_count = group.loc[start_idx:end_idx, 'word'].apply(count_syllables).sum()
                sps = syllable_count / duration if duration > 0 else 0
            else:
                duration = None
                sps = None
            return pd.Series({'duration': duration, 'syllables_per_second': sps})

        # Assign a unique sentence group identifier. Assumes each sentence starts with position == 1.
        df['sentence_group'] = (df['position'] == 1).cumsum()

        # Compute metrics per sentence.
        sentence_metrics = df.groupby('sentence_group').apply(compute_sentence_metrics).reset_index()
        overall_mean = sentence_metrics['syllables_per_second'].dropna().mean()
        # Replace None values with overall mean.
        sentence_metrics['syllables_per_second'] = sentence_metrics['syllables_per_second'].fillna(overall_mean)

        # Merge the syllables_per_second data back into the main DataFrame.
        if 'syllables_per_second' in df.columns:
            df.drop('syllables_per_second', axis=1, inplace=True)
        df = df.merge(sentence_metrics[['sentence_group', 'syllables_per_second']], on='sentence_group', how='left')

        CENTER = 5.2  # Scientifically supported average conversational speech rate
        SCALE = 1.2   # Approximated standard deviation from evidence-based research

        def sigmoid_normalize_sps(value):
            return 1 / (1 + np.exp(-(value - CENTER) / SCALE))

        # Apply sigmoid normalization.
        df['speed'] = df['syllables_per_second'].apply(sigmoid_normalize_sps)
        df['speed'] = df['speed'].clip(0, 1)  # Explicitly clip to [0,1] for numerical safety

        # Remove the temporary syllables_per_second column.
        return df.drop('syllables_per_second', axis=1)

    def get_sentence_speech_volume(self, df, name) -> pd.DataFrame:
        """
        Process sentence volume metrics and add a 'speech_volume' column to the DataFrame.
        
        The function:
        - Loads the vocals audio file from data_dir using the provided name.
        - Computes the volume (root mean square) for each sentence based on valid start/end times.
        - Normalizes the volume with respect to the mean (with a cap at twice the mean).
        - Updates DataFrame with a 'speech_volume' column.
        
        Args:
            name (str): Base name to locate the vocals file (file must be data/{name}_vocals.wav).
            df (pd.DataFrame): DataFrame containing columns: 'sentence_group', 'start_time', 
                            'end_time', and optionally 'sentence'.
            data_dir (str): Directory where the vocals audio file is stored.
        
        Returns:
            pd.DataFrame: Updated DataFrame with a new 'speech_volume' column.
        """

        # Ensure vocals are loaded only once.
        self.load_vocals(name)
        vocals_audio = self.vocals_audio
        sr_vocals = self.sr_vocals

        def calculate_segment_volume(audio, sr, start_time, end_time):
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            # Ensure indices are within valid bounds
            start_sample = max(0, min(start_sample, len(audio) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(audio)))
            segment = audio[start_sample:end_sample]
            return np.sqrt(np.mean(np.square(segment))) if len(segment) > 0 else 0.0

        sentence_volumes = {}
        # Compute volume for each sentence_group
        for group_id, group_df in df.groupby("sentence_group"):
            valid_start = group_df["start_time"].dropna()
            valid_end = group_df["end_time"].dropna()
            if not valid_start.empty and not valid_end.empty:
                start_time = valid_start.iloc[0]
                end_time = valid_end.iloc[-1]
                sentence_volume = calculate_segment_volume(vocals_audio, sr_vocals, start_time, end_time)
            else:
                sentence_volume = np.nan
            sentence_volumes[group_id] = sentence_volume
            sentence_text = group_df["sentence"].iloc[0] if "sentence" in group_df.columns else f"Group {group_id}"
            print(f"Sentence: {sentence_text[:30]}... Volume: {sentence_volume if not np.isnan(sentence_volume) else 'NaN'}")

        # Compute overall mean volume only from valid sentences
        valid_volumes = [vol for vol in sentence_volumes.values() if not np.isnan(vol)]
        mean_volume = np.mean(valid_volumes) if valid_volumes else 0.0
        print(f"Mean sentence volume (valid only): {mean_volume:.6f}")

        # Compute volume ratios for each sentence group (default to 1 if missing or mean_volume is 0)
        sentence_volume_ratios = {group_id: (vol / mean_volume if (not np.isnan(vol) and mean_volume > 0) else 1)
                                for group_id, vol in sentence_volumes.items()}

        # Map the sentence volume ratio to each row in the DataFrame
        df["volume_ratio"] = df["sentence_group"].map(sentence_volume_ratios)

        unweighted_mean_ratio = np.mean(list(sentence_volume_ratios.values()))
        print(f"Unweighted mean volume ratio: {unweighted_mean_ratio:.6f}")

        # Normalize volume ratio to the range [0, 1]
        min_volume_ratio = 0  # minimum reference
        max_volume_ratio = 2  # reference for twice as loud as the mean

        df['speech_volume'] = np.clip((df['volume_ratio'] - min_volume_ratio) / (max_volume_ratio - min_volume_ratio), 0, 1)
        df = df.drop('volume_ratio', axis=1)

        print("Updated DataFrame with 'speech_volume':")
        print(df)
        return df

    def get_word_ambient_volume(self, df, name) -> pd.DataFrame:
        """
        Compute per-word ambient volume ratios from vocals and instrumental tracks 
        and add a normalized "ambient_volume" column to the DataFrame.
        
        Args:
            name (str): Base name used to form the filename
            df (pd.DataFrame): DataFrame containing word info and timing info.
                Must include columns: "word", "start_time", "end_time".
            vocals_audio (np.ndarray): Audio data for vocals.
            sr_vocals (int): Sample rate for the vocals audio.
            
        Returns:
            pd.DataFrame: Updated DataFrame with an added "ambient_volume" column.
        """

        # Ensure vocals are loaded.
        self.load_vocals(name)
        vocals_audio = self.vocals_audio
        sr_vocals = self.sr_vocals

        # Load the instrumental (no_vocals) audio file.
        no_vocals_path = os.path.join(f"{name}_no_vocals.wav")
        print("Loading instrumental audio file from:", no_vocals_path)
        no_vocals_audio, sr_no_vocals = librosa.load(no_vocals_path, sr=None)

        # Ensure sample rates match
        if sr_vocals != sr_no_vocals:
            print(f"Warning: Sample rates differ (vocals: {sr_vocals}, instrumental: {sr_no_vocals})")
            print("Resampling instrumental track to match vocals sample rate...")
            no_vocals_audio = librosa.resample(no_vocals_audio, orig_sr=sr_no_vocals, target_sr=sr_vocals)
            sr_no_vocals = sr_vocals

        sr = sr_vocals

        def calculate_segment_volume(audio, sr, start_time, end_time):
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            # Ensure indices are within bounds
            start_sample = max(0, min(start_sample, len(audio) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(audio)))
            segment = audio[start_sample:end_sample]
            return np.sqrt(np.mean(np.square(segment))) if len(segment) > 0 else 0.0

        def get_fallback_times(df, idx, epsilon=0.01):
            """
            Returns a tuple (start_time, end_time) for the word at index idx using
            fallback values if needed from previous or next available words.
            """
            word = df.loc[idx]
            start_time = word["start_time"]
            end_time = word["end_time"]

            # Fallback for missing start_time: search backward for available end_time.
            if pd.isna(start_time):
                for j in range(idx - 1, -1, -1):
                    if pd.notna(df.loc[j, "end_time"]):
                        start_time = df.loc[j, "end_time"]
                        break
            # Fallback for missing end_time: search forward for available start_time.
            if pd.isna(end_time):
                for j in range(idx + 1, len(df)):
                    if pd.notna(df.loc[j, "start_time"]):
                        end_time = df.loc[j, "start_time"]
                        break

            # Final fallbacks if still missing.
            if pd.isna(start_time) and pd.notna(end_time):
                start_time = end_time - epsilon
            if pd.isna(end_time) and pd.notna(start_time):
                end_time = start_time + epsilon
            if pd.isna(start_time) and pd.isna(end_time):
                start_time, end_time = 0, epsilon

            return start_time, end_time

        def log_normalize(x, C=4):
            return np.clip(min(np.log(1 + x) / np.log(1 + C), 1), 0, 1)

        word_volumes = {}
        # Iterate over each word in the DataFrame.
        for idx, word in df.iterrows():
            start_time, end_time = get_fallback_times(df, idx)
            # Calculate volumes for both vocals and instrumental tracks.
            vocals_volume = calculate_segment_volume(vocals_audio, sr, start_time, end_time)
            no_vocals_volume = calculate_segment_volume(no_vocals_audio, sr, start_time, end_time)
            # Compute the ratio; if instrumental volume is zero, use a default high value.
            ratio = vocals_volume / no_vocals_volume if no_vocals_volume > 0 else 10.0

            word_volumes[idx] = {
                "vocals_volume": vocals_volume,
                "no_vocals_volume": no_vocals_volume,
                "ratio": ratio
            }

        # Apply logarithmic normalization to emphasize values below 1.
        normalized_ratios = {}
        for idx, data in word_volumes.items():
            normalized = 1 - log_normalize(data["ratio"])
            normalized_ratios[idx] = normalized

        # Map the normalized value to each row in the DataFrame.
        df["ambient_volume"] = [normalized_ratios[idx] for idx in df.index]
        return df
    
    
    def get_word_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate normalized word frequency scores using the Zipf scale.
        
        Args:
            df (pd.DataFrame): DataFrame containing a 'word' column
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'frequency' column
        """
        # Create cache for word frequencies if not already present
        if not hasattr(self, 'frequency_cache'):
            self.frequency_cache = {}
        
        # Get unique words that aren't in cache
        unique_words = set(df['word'].unique()) - set(self.frequency_cache.keys())
        
        # Calculate frequencies for new words in batch
        for word in unique_words:
            zipf_score = zipf_frequency(word.lower(), 'en')
            # Normalize and clip in one step
            self.frequency_cache[word] = min(max(zipf_score / 8, 0), 1)
        
        # Create result DataFrame efficiently using map
        df_result = df.copy()
        df_result['frequency'] = df_result['word'].map(self.frequency_cache)
        
        return df_result
