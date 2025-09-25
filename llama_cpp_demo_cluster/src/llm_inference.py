import os
import json
import pandas as pd
import requests

# Get Path of Python Script
def get_script_path():
    """Get the absolute path of the script."""
    return os.path.abspath('.')


# Function to load dataset
def load_data(csv_path: str, text_column: str = "text"):
    """Load dataset from CSV for inference, ensuring consistent return types."""
    df = pd.read_csv(csv_path)
    
    if text_column not in df.columns:
        raise ValueError(f"CSV file must contain a '{text_column}' column")
    
    # Remove NA and empty string values in the text column
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.strip() != '']

    # Ensure 'prediction' column exists
    if "prediction" not in df.columns:
        df["prediction"] = None  # Or np.nan if preferred

    return df

# Checkpoint Saving
def save_checkpoint(df, classified_path):
    """Writes data to a temporary file first, then atomically replaces the classified file."""
    temp_path = classified_path + ".tmp"
    backup_path = classified_path + ".old"

    # Write to temp file
    df.to_csv(temp_path, index=False)
    print(f"Saved temporary results to {temp_path}")

    # Ensure the temp file is fully written, then swap files
    if os.path.exists(classified_path):
        os.rename(classified_path, backup_path)  # Move old file to backup
    os.rename(temp_path, classified_path)  # Promote temp file to final file

    # Remove the old backup file
    if os.path.exists(backup_path):
        os.remove(backup_path)
    
    print(f"Successfully updated {classified_path}.")


############################
# Function to call llama.cpp server
############################
LLAMA_SERVER_URL = "http://localhost:8080/completion"

def call_llama_server(prompt: str, max_tokens: int = 2, temperature: float = 0.0, top_p: float = 0.9):
    """
    Calls the /completion endpoint of the llama.cpp server running at LLAMA_SERVER_URL.
    We pass 'prompt' and some generation parameters in JSON:
      {
        "prompt": "...",
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": top_p
      }

    Returns the 'content' field from the server's JSON response.
    """
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,   # synonyms: max_tokens / n_predict
        "temperature": temperature,
        "top_p": top_p
    }
    try:
        response = requests.post(LLAMA_SERVER_URL, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with llama.cpp server: {e}")
        return ""

    # Server should return JSON with a "content" field
    try:
        data = response.json()
        return data.get("content", "")
    except json.JSONDecodeError:
        # If for some reason it doesn't parse as JSON, return raw text
        return response.text

############################
# Quick test of the server
############################
script_path = get_script_path()
print(f"Script path is: {script_path}")

prompt_test = (
    "Sie werden Texte von Usern aus dem Internet lesen und bestimmen, ob es sich um Hate Speech handelt oder nicht. Es gibt keine allgemeingültige Definition von Hate Speech. Eine weit verbreitete und auch von den Vereinten Nationen verwendete Definition versteht unter Hate-Speech den Gebrauch von Sprache, durch den eine Person oder eine Gruppe aufgrund ihrer Identität – z.B. aufgrund ihres Geschlechts, Alters, ihrer Sexualität, Religion, Nationalität, Hautfarbe oder Herkunft, geistigen oder körperlichen Beeinträchtigung – angegriffen oder abgewertet wird. Lesen Sie den Text. Wenn ein Text keine Hate Speech enthält, antworte mit '0'. Wenn ein Kommentar Hate Speech enthält, antworte mit '1'"
    "Answer format: 0/1"
    "This is the text you need to decide if it contains hate speech or not:  Ich hasse die Christen, die sind alle Dumm und Blöd."
    "the correct answer is "
)
test_output = call_llama_server(
    prompt=prompt_test,
    max_tokens=2,
    temperature=0.0,
    top_p=0.9
)

print("=== Test Output ===")
print(json.dumps(test_output, indent=4, ensure_ascii=False))

############################
# Example run_inference
############################
def run_inference(texts, verbose=False):
    """
    Loop over a list of texts, build a prompt for each,
    call the server, and return predictions in a list.
    """
    predictions = []
    for text_item in texts:
        # Build the prompt (similar to your hate-speech logic)
        prompt = (
            "Sie werden Texte von Usern aus dem Internet lesen und bestimmen, ob es sich um Hate Speech handelt oder nicht. "
            "Es gibt keine allgemeingültige Definition von Hate Speech. Eine weit verbreitete und auch von den Vereinten Nationen verwendete Definition versteht unter Hate-Speech den Gebrauch von Sprache, durch den eine Person oder eine Gruppe aufgrund ihrer Identität – z.B. aufgrund ihres Geschlechts, Alters, ihrer Sexualität, Religion, Nationalität, Hautfarbe oder Herkunft, geistigen oder körperlichen Beeinträchtigung – angegriffen oder abgewertet wird. Lesen Sie den Text. Wenn ein Text keine Hate Speech enthält, antworte mit '0'. Wenn ein Kommentar Hate Speech enthält, antworte mit '1'. "
            "The Answer format: 0/1. "
            f"This is the text you need to decide if it contains hate speech or not: {text_item} "
            "the correct answer is "
        )

        # Optionally print the prompt for debugging
        #print("\n=== Final Prompt Sent to Model ===")
        #print(prompt)
        #print("===")

        output = call_llama_server(
            prompt=prompt,
            max_tokens=2,   # or however many tokens you need for your task ()for classification 1 or 2 tokens should be enough)
            temperature=0.0, # deterministic output for classification (temp = 0) if you want more randomness, increase this
            top_p=0.9 
        )

        # The server typically returns a string like "0" or "1" or more.
        # We just take the first character if we expect '0' or '1'.
        # This can be adjusted based on your specific needs.
        print("===")
        print(json.dumps(output, indent=4, ensure_ascii=False))
        prediction = output.strip()[:1]
        predictions.append(prediction)

    return predictions

############################
# Example main() logic
############################
def main(text_column="tweet_original", batch_size=10000):
    script_path = get_script_path()

    # Paths
    csv_path = os.path.join(script_path, "../../data/tweets_for_llm_classification.csv")
    classified_path = os.path.join(script_path, "../../data/tweets_for_llm_classified.csv")

    # Load data and check if job needs to be resumed
    if os.path.exists(classified_path):
        print("Found existing classification file. Resuming from partial results...")
        df = load_data(classified_path, text_column=text_column)
    else:
        print("No existing classification file found. Loading from original CSV...")
        df = load_data(csv_path, text_column=text_column)


    #  Identify rows where prediction is missing needed in case of resuming
    missing_indices = df[df["prediction"].isnull()].index.tolist()
    
    # If no missing rows, exit early
    if not missing_indices:
        print("All rows are already classified. No inference needed.")
        return

    print(f"Total rows without predictions: {len(missing_indices)}")

    # Batch processing
    for start_idx in range(0, len(missing_indices), batch_size):
        end_idx = min(start_idx + batch_size, len(missing_indices))
        batch_ids = missing_indices[start_idx:end_idx]

        batch_texts = df.loc[batch_ids, text_column].tolist()

        # Call run_inference (which calls the server) for this batch
        predictions = run_inference(batch_texts, verbose=False)

        # Store predictions
        df.loc[batch_ids, 'prediction'] = predictions

        # Save partial results
        save_checkpoint(df, classified_path)
        df.to_csv(classified_path, index=False)
        print(f"Processed batch #{start_idx // batch_size + 1} ({end_idx - start_idx} rows). "
              f"Results saved to {classified_path}.")

    print("================================")
    print("Inference completed. Final predictions are saved.")
    print("================================")


############################
# Entry point
############################
# You can adjust text_column and batch_size as needed (advice smaller batch size will lead to more IO on Storage which can impact performance)
if __name__ == "__main__":
    main(text_column="tweet_original", batch_size=25000)