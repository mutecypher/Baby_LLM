##from memory_usage import log_memory_usage
##from concurrent.futures import ProcessPoolExecutor
##import logging
##import gc
##import psutil

def process_file(filename, tokenizer):
    file_path = os.path.join(gutenberg_dir, filename)
    cleaned_file_path = os.path.join(cleaned_dir, f"{filename}.cleaned.txt")
    logging.debug(f"Processing file: {filename}")
    
    if not os.path.isfile(file_path):
        logging.warning(f"File not found: {filename}")
        return ""
    if os.path.getsize(file_path) > 10 * 1024 * 1024:
        logging.info(f"Skipping {filename}: File too large (>10MB)")
        return ""
    
    if os.path.exists(cleaned_file_path):
        with open(cleaned_file_path, "r", encoding="utf-8") as f:
            return f.read() + "\n\n"
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        logging.info(f"Raw text length for {filename}: {len(raw_text)}")
        stripped_text = simple_strip_headers(raw_text, filename=filename)
        logging.info(f"Stripped text length: {len(stripped_text)}")
        preprocessed_text = preprocess_text(stripped_text)
        logging.info(f"Preprocessed text length: {len(preprocessed_text)}")
        cleaned_text = enhanced_clean_text(preprocessed_text)
        logging.info(f"Cleaned text length: {len(cleaned_text)}")
        
        if not cleaned_text.strip():
            logging.warning(f"Empty cleaned text for {filename}")
            return ""
        
        if len(cleaned_text) > 10:
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


def process_file_batch(filenames, tokenizer, batch_size=32):
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(filenames), batch_size):
            batch = filenames[i:i + batch_size]
            future_to_file = {executor.submit(process_file, fname, tokenizer): fname for fname in batch}
            for future in future_to_file:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.error(f"Error processing {future_to_file[future]}: {str(e)}")
            gc.collect()
            log_memory_usage()
    return results
