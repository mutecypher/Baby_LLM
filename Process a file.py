import os
import logging
import re
import numpy as np

logging.basicConfig(filename='test_7142_simple.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
gutenberg_dir = os.path.expanduser("~/Baby_LLM/data/gutenberg")
cleaned_dir = os.path.expanduser("~/Baby_LLM/data/cleaned")
os.makedirs(cleaned_dir, exist_ok=True)
filename = "7142.txt"

# Minimal patterns
patterns = [
    (re.compile(r'\n{3,}', re.MULTILINE), '\n\n'),
]

def preprocess_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def enhanced_clean_text(raw_text):
    logging.debug(f"Raw text length: {len(raw_text)}, sample: {raw_text[:200]}")
    text = raw_text
    for pattern, repl in patterns:
        matches = pattern.findall(text)
        if matches:
            logging.debug(f"Pattern {pattern.pattern} matched {len(matches)} times, sample: {matches[:5]}")
        text = pattern.sub(repl, text)
    logging.debug(f"Final cleaned text: length={len(text)}, sample={text[:200]}")
    return text.strip()

def simple_strip_headers(text, include_dedication=True):
    # Split text into lines
    lines = text.splitlines()
    total_lines = len(lines)
    logging.debug(f"Total lines in text: {total_lines}")

    # Find first occurrence of "Gutenberg"
    first_gutenberg_idx = -1
    for i, line in enumerate(lines):
        if "Gutenberg" in line:
            first_gutenberg_idx = i
            break
    if first_gutenberg_idx == -1:
        logging.warning("No 'Gutenberg' found in text; returning original text")
        return text

    # Skip first Gutenberg line + next 20 lines
    start_idx = first_gutenberg_idx + 20 + 1
    if start_idx >= total_lines:
        logging.warning(f"Start index {start_idx} exceeds total lines {total_lines}; adjusting to end")
        start_idx = total_lines
    logging.debug(f"First 'Gutenberg' found at line {first_gutenberg_idx}; starting text at line {start_idx}")
    logging.debug(f"Lines skipped at start (first 10): {lines[first_gutenberg_idx:min(first_gutenberg_idx+10, start_idx)]}")

    # Find last occurrence of "Gutenberg"
    last_gutenberg_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Gutenberg" in lines[i]:
            last_gutenberg_idx = i
            break
    if last_gutenberg_idx == -1:
        logging.warning("No last 'Gutenberg' found; using end of text")
        last_gutenberg_idx = total_lines

    # Remove last Gutenberg line + previous 20 lines
    end_idx = last_gutenberg_idx - 20
    if end_idx < start_idx:
        logging.warning(f"End index {end_idx} is before start index {start_idx}; adjusting to start")
        end_idx = start_idx
    logging.debug(f"Last 'Gutenberg' found at line {last_gutenberg_idx}; ending text at line {end_idx}")
    logging.debug(f"Lines skipped at end (last 10): {lines[max(0, end_idx-10):last_gutenberg_idx+1]}")

    # Extract text between start_idx and end_idx
    stripped_lines = lines[start_idx:end_idx]
    
    # Remove lines containing metadata terms
    metadata_terms = ["author", "translator", "language", "utf-8", "produced by", "gutenberg", "translated by"]
    metadata_patterns = [
        re.compile(r'^\s*\*{3}\s*START OF.*GUTENBERG.*$', re.IGNORECASE),  # Match *** START ...
        re.compile(r'^\s*THE (HISTORY OF THE PELOPONNESIAN WAR|MUSE OF THE DEPARTMENT)\s*$', re.IGNORECASE),  # Match titles
        re.compile(r'^\s*by (thucydides|honore de balzac)\s*$', re.IGNORECASE),  # Match author lines
        re.compile(r'^\s*DEDICATION\s*$', re.IGNORECASE),  # Match dedication header
        re.compile(r'^\s*(PREFACE|INTRODUCTION|FOREWORD)\s*$', re.IGNORECASE),  # Match other front matter
    ]
    cleaned_lines = []
    removed_lines = []
    narrative_started = False
    dedication_seen = False
    dedication_lines = []
    lines_since_start = 0
    max_front_matter_lines = 50  # Fallback to prevent discarding all text

    for i, line in enumerate(stripped_lines):
        lines_since_start += 1
        # Detect dedication header
        if re.search(r'^\s*DEDICATION\s*$', line, re.IGNORECASE):
            dedication_seen = True
            removed_lines.append(line)
            continue
        # Collect dedication lines
        if dedication_seen and not narrative_started and len(line.strip()) > 0:
            if re.match(r'^\s*(to |my dear|dear|sir|madam|friend)', line, re.IGNORECASE) or len(dedication_lines) > 0:
                dedication_lines.append(line)
                continue
        # Start narrative after "CHAPTER", "Thucydides, an Athenian", or prose after dedication
        if (re.search(r'^(CHAPTER|Thucydides, an Athenian)', line, re.IGNORECASE) or
            (dedication_seen and len(line.strip()) > 50 and not re.match(r'^\s*(to |my dear|dear|sir|madam|friend)', line, re.IGNORECASE))):
            narrative_started = True
            logging.debug(f"Narrative started at line {start_idx + i}: {line[:100]}")
        # Fallback: Start narrative after max_front_matter_lines
        if not narrative_started and lines_since_start > max_front_matter_lines:
            narrative_started = True
            logging.debug(f"Forced narrative start at line {start_idx + i} due to max_front_matter_lines: {line[:100]}")
        # Skip lines until narrative starts, unless part of dedication
        if not narrative_started:
            removed_lines.append(line)
            continue
        # Check for metadata terms or patterns
        if (any(term.lower() in line.lower() for term in metadata_terms) or
            any(pattern.search(line) for pattern in metadata_patterns)):
            removed_lines.append(line)
        else:
            if include_dedication and dedication_seen and not cleaned_lines:
                cleaned_lines.extend(dedication_lines)
                logging.debug(f"Included {len(dedication_lines)} dedication lines: {dedication_lines[:5]}")
            cleaned_lines.append(line)
    if removed_lines:
        logging.debug(f"Removed {len(removed_lines)} lines containing metadata or front matter: {removed_lines[:10]}")
    
    stripped_text = "\n".join(cleaned_lines).strip()
    logging.debug(f"Text after simple_strip_headers: length={len(stripped_text)}, sample={stripped_text[:200]}")
    if not stripped_text.strip():
        logging.warning(f"Empty output from simple_strip_headers; sample of stripped_lines: {stripped_lines[:10]}")
    return stripped_text

def process_file(filename):
    file_path = os.path.join(gutenberg_dir, filename)
    cleaned_file_path = os.path.join(cleaned_dir, f"{filename}.cleaned.txt")
    logging.debug(f"Processing file: {filename}, path: {file_path}")
    if not os.path.isfile(file_path):
        logging.warning(f"File not found: {filename}")
        return ""
    if os.path.getsize(file_path) > 100 * 1024 * 1024:
        logging.info(f"Skipping {filename}: File too large (>100MB)")
        return ""
    try:
        logging.info(f"Reading raw text for {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        logging.debug(f"Raw text length for {filename}: {len(raw_text)}")
        stripped_text = simple_strip_headers(raw_text, include_dedication=True)
        with open(os.path.join(cleaned_dir, f"{filename}.stripped.txt"), "w", encoding="utf-8") as f:
            f.write(stripped_text)
        preprocessed_text = preprocess_text(stripped_text)
        logging.debug(f"Preprocessed text length for {filename}: {len(preprocessed_text)}, sample: {preprocessed_text[:200]}")
        with open(os.path.join(cleaned_dir, f"{filename}.preprocessed.txt"), "w", encoding="utf-8") as f:
            f.write(preprocessed_text)
        cleaned_text = enhanced_clean_text(preprocessed_text)
        logging.debug(f"Cleaned text length for {filename}: {len(cleaned_text)}, sample: {cleaned_text[:200]}")
        with open(os.path.join(cleaned_dir, f"{filename}.cleaned.debug.txt"), "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        if not cleaned_text.strip():
            logging.warning(f"Empty cleaned text for {filename}")
            return ""
        if len(cleaned_text) > 500:
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            logging.info(f"Saved cleaned file: {filename}")
            return cleaned_text + "\n\n"
        else:
            logging.info(f"Skipping {filename}: Too short (length={len(cleaned_text)})")
            return ""
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return ""

result = process_file(filename)
logging.info(f"Result for {filename}: length={len(result)}, sample={result[:200]}")