import csv
import re
from indexing import preprocess_and_index

def process_column_content(s):
    """Clean and strip unwanted characters."""
    return re.sub(r'\(.*?\)|\".*?\"|\[.*?\]', '', s).strip().lower()


def extract_terms_and_ids_from_csv(input_filename):
    """
    Extract terms and IDs from the provided CSV file into a dictionary.

    Args:
        input_filename (str): Path to the CSV file.

    Returns:
        dict: Dictionary where keys are terms and values are IDs.
    """
    term_to_id = {}
    with open(input_filename, "r") as infile:
        reader = csv.reader(infile, delimiter="|")
        for row in reader:
            if "authority" not in row[3]:
                term_name = process_column_content(row[1])
                if not term_name or len(term_name) <= 1:
                    continue
                term_id = row[0].strip()
                term_to_id[term_name] = term_id
    return term_to_id


if __name__ == "__main__":
    # Paths for input and output files
    INPUT_FILENAME = "/home/stirunag/work/github/source_data/knowledge_base/taxdump/names.dmp"
    OUTPUT_PICKLE_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/NCBI_terms.pkl"
    OUTPUT_LIST = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/NCBI_list.txt"
    FAISS_INDEX_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/NCBI_terms.index"
    MODEL_PATH = "/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model"

    # Extract terms and IDs
    term_id_dict = extract_terms_and_ids_from_csv(INPUT_FILENAME)

    # Preprocess and Index the terms
    preprocess_and_index(
        term_id_dict,
        OUTPUT_PICKLE_FILENAME,
        OUTPUT_LIST,
        FAISS_INDEX_FILENAME,
        MODEL_PATH
    )