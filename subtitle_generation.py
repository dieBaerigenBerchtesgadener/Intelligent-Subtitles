import pandas as pd
from utils import remove_punctuation_not_between_letters, seconds_to_srt_timestamp, clean_token

def count_tokens(block_lines: list[str]) -> int:
    count = 0
    for line in block_lines[2:]:
        words = line.split()
        count += len([w for w in words if clean_token(w)])
    return count

def merge_and_clean_placeholders(words: list[str], placeholder: str) -> list[str]:
    merged_words = []
    for i, w in enumerate(words):
        if w != placeholder:
            if merged_words and merged_words[-1] == placeholder:
                merged_words[-1] = '•'  # Replace placeholder with a single dot
            merged_words.append(w)
        elif not merged_words or (i > 0 and words[i-1] != placeholder):
            merged_words.append(w)
    
    # Remove leading and trailing placeholders
    while merged_words and merged_words[0] == placeholder:
        merged_words.pop(0)
    while merged_words and merged_words[-1] == placeholder:
        merged_words.pop()
    
    return merged_words

def create_srt_file(
    srt_lines: list[str],
    df: pd.DataFrame,
    new_srt_file: str,
    original_timesteps: bool = False,
    languages: list[str] = ["en", "de"],
    english_level: float = 0.0  # New parameter for user's English level (0.0-1.0)
) -> None:
    placeholder = '•'
    display_flags = df["display"].tolist()
    df_index = 0
    previous_end_time = 0

    with open(new_srt_file, 'w', encoding='utf-8') as outfile:
        subtitle_block = []
        for line in srt_lines:
            if line.strip() == '':
                if subtitle_block:
                    previous_end_time = _write_block(subtitle_block, display_flags, df_index, df,
                                 outfile, placeholder, languages, original_timesteps, previous_end_time,
                                 english_level)  # Pass english_level parameter
                    df_index += count_tokens(subtitle_block)
                    subtitle_block = []
                outfile.write('\n')
            else:
                subtitle_block.append(line)

        if subtitle_block:
            _write_block(subtitle_block, display_flags, df_index, df,
                         outfile, placeholder, languages, original_timesteps, previous_end_time,
                         english_level)  # Pass english_level parameter

def _write_block(block_lines: list[str],
                 display_flags: list[bool],
                 start_idx: int,
                 df: pd.DataFrame,
                 outfile,
                 placeholder: str,
                 languages: list[str],
                 original_timesteps: bool,
                 previous_end_time: float,
                 english_level: float = 0.0):  # Add english_level parameter
    if len(block_lines) < 2:
        return previous_end_time

    subtitle_number = block_lines[0].strip()
    timestamp_line = block_lines[1].strip()
    text_lines = block_lines[2:]
    token_count = count_tokens(block_lines)

    # Identify the first and last displayed indices with valid start_time/end_time.
    first_displayed_idx = None
    last_displayed_idx = None
    for i in range(token_count):
        current_idx = start_idx + i
        if current_idx < len(display_flags) and display_flags[current_idx]:
            if first_displayed_idx is None and pd.notna(df.loc[current_idx, 'start_time']):
                first_displayed_idx = current_idx
            if pd.notna(df.loc[current_idx, 'end_time']):
                last_displayed_idx = current_idx

    try:
        original_start, original_end = timestamp_line.split(' --> ')
    except ValueError:
        print(f"Warning: Invalid timestamp format in block {subtitle_number}.")
        original_start, original_end = "00:00:00,000", "00:00:00,000"

    # Determine new start time using the first valid start_time value.
    if not original_timesteps and first_displayed_idx is not None:
        start_val = df.loc[first_displayed_idx, 'start_time']
        if pd.notna(start_val):
            new_start_time = seconds_to_srt_timestamp(start_val)
            new_start_seconds = srt_timestamp_to_seconds(new_start_time)
            # In case the new start is before the previous block's end, adjust it.
            if new_start_seconds < previous_end_time:
                new_start_time = seconds_to_srt_timestamp(previous_end_time)
        else:
            new_start_time = original_start
    else:
        new_start_time = original_start

    # Determine new end time using the last valid end_time value.
    if not original_timesteps and last_displayed_idx is not None:
        end_val = df.loc[last_displayed_idx, 'end_time']
        if pd.notna(end_val):
            new_end_time = seconds_to_srt_timestamp(end_val)
        else:
            new_end_time = original_end
    else:
        new_end_time = original_end

    # Assemble the updated timestamp line.
    timestamp_line = f"{new_start_time} --> {new_end_time}"

    # Process the text lines into final subtitle text per language.
    filtered_tokens_per_lang = {lang: [] for lang in languages}
    df_index = start_idx
    skip_occurred = {lang: False for lang in languages}
    for line in text_lines:
        words = line.split()
        for w in words:
            t_clean = clean_token(w)
            if t_clean is None:
                # We simply ignore tokens that cannot be cleaned.
                continue
                
            if df_index < len(display_flags) and display_flags[df_index]:
                # This word should be displayed
                word_complexity = df.loc[df_index, "word_complexity"]
                
                for lang in languages:
                    if skip_occurred[lang]:
                        filtered_tokens_per_lang[lang].append(placeholder)
                        skip_occurred[lang] = False
                    
                    # For English, always show the word
                    if lang == "en":
                        filtered_tokens_per_lang[lang].append(w)
                    # For other languages, check complexity
                    elif word_complexity > english_level:
                        # Show translation for complex words
                        filtered_tokens_per_lang[lang].append(df.loc[df_index, "translation"])
                    else:
                        # Skip translation for simple words
                        skip_occurred[lang] = True
            else:
                # This word should not be displayed
                for lang in languages:
                    skip_occurred[lang] = True
                    
            df_index += 1

    # Merge token lists to create final text lines per language.
    for lang in languages:
        merged = merge_and_clean_placeholders(filtered_tokens_per_lang[lang], placeholder)
        merged_line = remove_punctuation_not_between_letters(" ".join(merged))
        filtered_tokens_per_lang[lang] = [merged_line] if merged_line else []

    # Write out the subtitle block for each language that has text.
    for lang in languages:
        text_lines_lang = filtered_tokens_per_lang[lang]
        if text_lines_lang:
            single_line = " ".join(text_lines_lang)
            outfile.write(f"{subtitle_number}\n")
            outfile.write(f"{timestamp_line}\n")
            outfile.write(f"{single_line}\n\n")

    # Update the previous_end_time based on new_end_time.
    try:
        new_end_seconds = srt_timestamp_to_seconds(new_end_time)
    except Exception:
        new_end_seconds = previous_end_time

    return max(previous_end_time, new_end_seconds)

def srt_timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.replace(',', '.').split(':')
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
