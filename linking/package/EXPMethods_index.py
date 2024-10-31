import string
from bs4 import BeautifulSoup
from indexing import preprocess_and_index

import re
import string
from bs4 import BeautifulSoup

# List of phrases to remove (converted to lowercase)
phrases_to_remove = [
    # Add specific phrases to remove, e.g., 'assay', 'experiment'
]

# Replacement patterns (e.g., regex patterns to replace or remove)
replacement_patterns = [
    # Add regex patterns if needed, e.g., r'\bseq\b' to remove 'seq'

]


def clean_and_modify_term(term):
    """
    Clean and modify a term by removing specified phrases, patterns, and punctuation.

    Args:
        term (str): The term to clean.

    Returns:
        str: The cleaned and modified term.
    """
    # Convert term to lowercase
    term = term.lower()

    # Remove specified phrases using regex
    for phrase in phrases_to_remove:
        term = re.sub(rf'\b{re.escape(phrase)}\b', '', term, flags=re.IGNORECASE)

    # Replace specific patterns
    for pattern in replacement_patterns:
        term = re.sub(pattern, '', term)

    # Remove punctuation
    term = term.translate(str.maketrans('', '', string.punctuation))

    # Replace dashes with spaces and remove extra whitespace
    term = term.replace('-', ' ')
    term = ' '.join(term.split())  # Remove extra whitespace

    return term.lower()


def get_singular(term):
    """
    Convert plural term to singular by applying basic English pluralization rules.

    Args:
        term (str): The term to convert.

    Returns:
        str: The singular form of the term.
    """
    if term.endswith("es"):
        return term[:-2]
    elif term.endswith("s"):
        return term[:-1]
    return term


def extract_terms_and_ids_from_mwt(input_filename):
    """
    Extract terms and IDs from the provided MWT file into a dictionary,
    adding singular and cleaned versions of terms.

    Args:
        input_filename (str): Path to the MWT file.

    Returns:
        dict: Dictionary where keys are terms (with variations) and values are IDs.
    """
    term_to_id = {}

    with open(input_filename, "r", encoding="utf-8") as infile:
        content = infile.read()
        soup = BeautifulSoup(content, "xml")

    # Find all term elements
    terms = soup.find_all("t")
    for term in terms:
        term_id = term.get("p1").strip()
        term_name = term.text.lower()

        # Add the original term
        if term_name and len(term_name) > 1:
            term_to_id[term_name] = term_id

            # Generate the singular form if the term is plural
            singular_term = get_singular(term_name)
            if singular_term != term_name:
                term_to_id[singular_term] = term_id

            # Generate cleaned version and add it
            cleaned_term = clean_and_modify_term(term_name)
            if cleaned_term and cleaned_term not in term_to_id:
                term_to_id[cleaned_term] = term_id

            # Generate cleaned version of the singular term if applicable
            cleaned_singular_term = clean_and_modify_term(singular_term)
            if cleaned_singular_term and cleaned_singular_term not in term_to_id:
                term_to_id[cleaned_singular_term] = term_id

    return term_to_id


# Example usage
# term_to_id = extract_terms_and_ids_from_mwt("path_to_file.mwt")
# print(term_to_id)

if __name__ == "__main__":
    # Paths for input and output files
    INPUT_FILENAME = "/home/stirunag/work/github/source_data/knowledge_base/expMethods/experimentalMethods_Dict.mwt"
    OUTPUT_PICKLE_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/em_terms.pkl"
    OUTPUT_LIST = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/em_list.txt"
    FAISS_INDEX_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/em_terms.index"
    MODEL_PATH = "/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model"

    # Extract terms and IDs
    term_id_dict = extract_terms_and_ids_from_mwt(INPUT_FILENAME)

    # Preprocess and Index the terms
    preprocess_and_index(
        term_id_dict,
        OUTPUT_PICKLE_FILENAME,
        OUTPUT_LIST,
        FAISS_INDEX_FILENAME,
        MODEL_PATH
    )