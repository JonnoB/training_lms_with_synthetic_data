import re
import time
import pandas as pd
from tqdm import tqdm
import difflib


def training_prompt(sample, raw_ocr, clean_ocr, tokenizer):
    """
    This function converts a single example of the corrupted and corrected text into the correct prompt response format.
    It must be paired with a looping function to process all elements of the dataset
    """
    #bos_token = "<s>"
    instruction_message =  f"""You are an expert in recovering OCR text, please recover the below text from an 18th century British Newspaper. End the recovery with triple #"""
    input =  sample[raw_ocr]
    response = sample[clean_ocr]
    eos_token = tokenizer.eos_token

    full_prompt = ""
    #full_prompt += bos_token
    full_prompt +=  instruction_message
    full_prompt += "\n\n###Raw OCR###"
    full_prompt += "\n" + input
    full_prompt += "\n\n###Recovery###"
    full_prompt += "\n" + response
    full_prompt += "###"
    full_prompt += eos_token

    return {'full_prompt':full_prompt}

def inference_prompt(text):

    """
    Formats the text into the the correct structure for inference. Takes the corrupted OCR text string and outputs
    a text string with the prompt and formatting structure. 
    The prompt structure is based on the Unsloth approach see their documentation
    """

    instruction_message =  f"""You are an expert in recovering OCR text, please recover the below text from an 18th century British Newspaper. End the recovery with triple #"""

    full_prompt = ""
    full_prompt +=  instruction_message
    full_prompt += "\n\n###Raw OCR###"
    full_prompt += "\n" + text
    full_prompt += "\n\n###Recovery###"

    return full_prompt

def cleaning_prompt_formatter(sample, raw_ocr, tokenizer ):
    """ 
    #This function is not really being used any more
    """

    #bos_token = "<s>"
    instruction_message =  f"""You are an expert in recovering OCR text, please recover the below text from an 18th century British Newspaper. End the recovery with triple #"""
    input =  sample[raw_ocr]

    full_prompt = ""
    #full_prompt += bos_token
    full_prompt +=  instruction_message
    full_prompt += "\n\n###Raw OCR###"
    full_prompt += "\n" + input
    full_prompt += "\n\n###Recovery###"

    return {'full_prompt':full_prompt}


def compute_metric(row, metric, prediction_col, reference_col):
    try:
        # Preprocess the text: lowercasing and replacing line breaks with spaces
        prediction = re.sub(r'\s+', ' ', row[prediction_col].lower().strip())
        reference = re.sub(r'\s+', ' ', row[reference_col].lower().strip())

        # Ensure the inputs to metric.compute are lists of strings
        predictions = [prediction]
        references = [reference]
        return metric.compute(predictions=predictions, references=references)
    except KeyError as e:
        print(f"KeyError: {e} in row: {row}")
        return None
    except Exception as e:
        print(f"Error: {e} in row: {row}")
        return None



def infer_on_test_set_split(data, model, tokenizer, device="cuda", n=500, m=100):
    """
    Perform inference on a test set using a language model, with text splitting and stitching.

    Parameters:
    - data: A Hugging Face dataset or a list of samples, each containing 'full_prompt', 'token_length', 'file_name', 'article_text', 'raw_text'.
    - model: The pre-trained language model to be used for inference.
    - tokenizer: The tokenizer corresponding to the model.
    - device: The device to run the inference on ('cuda' for GPU, 'cpu' for CPU).
    - n: Number of characters per chunk for splitting the text.
    - m: Number of overlapping characters between consecutive chunks.

    Returns:
    - A pandas DataFrame containing the original data and the inferred results.
    """
    clocrc_texts = []
    tps_values = []
    print('Inferring on test set')

    # Wrap the data iterable with tqdm for a progress bar
    for sample in tqdm(data, desc="Processing Samples"):

        text = sample['ocr_text']

        # Split the text into chunks at the character level
        chunks = split_text(text, n, m)
        
        # Format prompts for all chunks
        formatted_prompts = [inference_prompt(chunk) for chunk in chunks]
        
        # Tokenize the entire list of chunks at once
        inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        start_time = time.time()

        # Generate outputs for all chunks at once
        output = model.generate(**inputs, max_new_tokens=max(sample['ocr_tokens'] for _ in chunks), pad_token_id=tokenizer.eos_token_id)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Decode all outputs simultaneously
        inferred_chunks = [tokenizer.decode(o, skip_special_tokens=False) for o in output]
        # Clean the chunk to remove the prompt format
        inferred_chunks = [ chunk.split('###Recovery###')[1].split('###')[0] for chunk in inferred_chunks ]
        # Stitch the inferred chunks back together
        stitched_text = stitch_text(inferred_chunks)

        # Calculate tokens generated per second
        num_generated_tokens = sum(len(tokenizer.encode(chunk)) for chunk in inferred_chunks) - sum(len(tokenizer.encode(chunk)) for chunk in chunks)
        tps = num_generated_tokens / elapsed_time

        # Append the result to the lists
        clocrc_texts.append(stitched_text)
        tps_values.append(tps)

    # Convert the Hugging Face dataset to a DataFrame
    result_df = pd.DataFrame(data)

    # Add the new columns with the inference results
    result_df['clocrc_text'] = clocrc_texts
    result_df['tps'] = tps_values

    return result_df



def infer_on_test_set(data, model, tokenizer, device="cuda"):
    """
    Perform inference on a test set using a language model.

    Parameters:
    - data: A Hugging Face dataset or a list of samples, each containing 'full_prompt', 'token_length', 'file_name', 'article_text', 'raw_text'.
    - model: The pre-trained language model to be used for inference.
    - tokenizer: The tokenizer corresponding to the model.
    - device: The device to run the inference on ('cuda' for GPU, 'cpu' for CPU).

    Returns:
    - A pandas DataFrame containing the original data and the inferred results.
    """
    clocrc_texts = []
    tps_values = []
    print('Inferring on test set')

    # Wrap the data iterable with tqdm for a progress bar
    for sample in tqdm(data, desc="Processing Samples"):

        text = sample['ocr_text']

        formatted_prompt = inference_prompt(text)

        inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)

        start_time = time.time()

        # Generate the output
        output = model.generate(**inputs, max_new_tokens=sample['ocr_tokens'], pad_token_id=tokenizer.eos_token_id)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Calculate tokens generated per second
        num_generated_tokens = output.shape[1] - inputs['input_ids'].shape[1]
        tps = num_generated_tokens / elapsed_time

        # Append the result to the lists
        clocrc_texts.append(tokenizer.decode(output[0], skip_special_tokens=False))
        tps_values.append(tps)

    # Convert the Hugging Face dataset to a DataFrame
    result_df = data.to_pandas()

    # Add the new columns with the inference results
    result_df['clocrc_text'] = clocrc_texts

    result_df['clocrc_text'] =  result_df['clocrc_text'].apply(lambda x: x.split('###Recovery###')[1].split('###')[0])

    result_df['tps'] = tps_values

    return result_df




def split_text(text: str, n: int, m: int):
    """
    Splits the input text into chunks of n characters with an overlap of m characters.

    Args:
    - text: The input text to split.
    - n: Number of characters in each chunk.
    - m: Number of overlapping characters between consecutive chunks.

    Returns:
    - A list of character-level text chunks.
    """
    chunks = []
    i = 0
    
    while i < len(text):
        chunk = text[i:i+n]
        chunks.append(chunk)
        i += (n - m)
    
    return chunks


def stitch_text(chunks: list):
    """
    Stitches the chunks of text back together using the Longest Common Subsequence (LCS) method.

    Args:
    - chunks: A list of strings, where each string is a chunk of recovered text.

    Returns:
    - The fully stitched text.
    """
    stitched_text = chunks[0]  # Start with the first chunk

    for i in range(1, len(chunks)):
        prev_chunk = stitched_text
        curr_chunk = chunks[i]
        
        # Use difflib to find the longest matching subsequence
        s = difflib.SequenceMatcher(None, prev_chunk, curr_chunk)
        match = s.find_longest_match(0, len(prev_chunk), 0, len(curr_chunk))
        
        # Append only the non-overlapping part of the current chunk's text
        if match.size > 0:
            stitched_text += curr_chunk[match.b + match.size:]
        else:
            stitched_text += curr_chunk

    return stitched_text
