from indexing import preprocess_and_index
import sys
import pandas as pd

def clean_term(term):
    """
    Clean the term by converting it to lowercase and removing single and double quotes.

    Args:
        term (str): The term to clean.

    Returns:
        str: The cleaned term.
    """
    # Convert to lowercase
    term = term.lower()

    # Remove single and double quotes
    term = term.replace("'", "").replace('"', '')

    return term

def extract_terms_and_ids_from_csv(input_filename):
    """
    Extract terms and IDs from the provided CSV file into a dictionary.

    Args:
        input_filename (str): Path to the CSV file.

    Returns:
        dict: Dictionary where keys are cleaned terms and values are term IDs.
    """
    term_to_id = {}

    # Read the CSV file using pandas
    try:
        df = pd.read_csv(input_filename, dtype=str)
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{input_filename}' is empty.", file=sys.stderr)
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Check if required columns exist
    if 'NAME' not in df.columns or 'URL' not in df.columns:
        print("Error: CSV file must contain 'NAME' and 'URL' columns.", file=sys.stderr)
        sys.exit(1)

    for index, row in df.iterrows():
        term_name = str(row['NAME']).strip()
        url = str(row['URL']).strip()

        if not term_name or not url:
            print(f"Warning: Missing 'NAME' or 'URL' in row {index + 1}. Skipping.", file=sys.stderr)
            continue  # Skip incomplete entries

        # Extract term_id from URL (last part after '/')
        try:
            term_id = url.rstrip('/').split('/')[-1]
            if not term_id.isdigit():
                print(f"Warning: term_id extracted from URL '{url}' is not numeric. Skipping row {index + 1}.", file=sys.stderr)
                continue
        except IndexError:
            print(f"Warning: URL '{url}' does not contain a valid term_id. Skipping row {index + 1}.", file=sys.stderr)
            continue

        # Clean the term
        cleaned_term = clean_term(term_name)

        if cleaned_term and len(cleaned_term) > 0:
            term_to_id[cleaned_term] = term_id
            print(term_id, cleaned_term)
        else:
            print(f"Warning: Cleaned term is empty for original term '{term_name}' in row {index + 1}. Skipping.", file=sys.stderr)

    return term_to_id

if __name__ == "__main__":
    # Paths for input and output files
    INPUT_FILENAME = "/home/stirunag/work/github/source_data/knowledge_base/Primer/primer_probebase.csv"
    OUTPUT_PICKLE_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/primer_terms.pkl"
    OUTPUT_LIST = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/primer_list.txt"
    FAISS_INDEX_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/primer_terms.index"
    MODEL_PATH = "/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model"

    # Extract terms and IDs
    term_id_dict = extract_terms_and_ids_from_csv(INPUT_FILENAME)

    if not term_id_dict:
        print("No terms extracted. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Preprocess and Index the terms
    preprocess_and_index(
        term_id_dict,
        OUTPUT_PICKLE_FILENAME,
        OUTPUT_LIST,
        FAISS_INDEX_FILENAME,
        MODEL_PATH
    )

    print(f"Indexing completed successfully. Outputs:")
    print(f" - Pickle file: {OUTPUT_PICKLE_FILENAME}")
    print(f" - List file: {OUTPUT_LIST}")
    print(f" - FAISS index: {FAISS_INDEX_FILENAME}")