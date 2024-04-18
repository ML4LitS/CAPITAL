from pronto import Ontology
from tqdm import tqdm
from indexing import preprocess_and_index

def extract_terms_and_ids_from_chebi(input_filename):
    """
    Extract terms and IDs from the provided ChEBI OWL ontology into a dictionary.

    Args:
        input_filename (str): Path to the OWL ontology file.

    Returns:
        dict: Dictionary where keys are terms and values are IDs.
    """
    chebi = Ontology(input_filename)
    term_to_id = {}

    print("Processing terms from ChEBI ontology...")
    for term in tqdm(chebi.terms(), total=len(chebi.terms()), desc="Extracting Terms"):
        chebi_id = term.id
        term_name = term.name

        if not term_name:
            continue

        term_name = term_name.lower()
        term_to_id[term_name] = chebi_id

    return term_to_id


if __name__ == "__main__":
    # Paths for input and output files
    INPUT_FILENAME = "/home/stirunag/work/github/source_data/knowledge_base/chebi/chebi.owl"
    OUTPUT_PICKLE_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/chebi_terms.pkl"
    OUTPUT_LIST = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/chebi_list.txt"
    FAISS_INDEX_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/chebi_terms.index"
    MODEL_PATH = "/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model"

    # Extract terms and IDs
    term_id_dict = extract_terms_and_ids_from_chebi(INPUT_FILENAME)

    # Preprocess and Index the terms
    preprocess_and_index(
        term_id_dict,
        OUTPUT_PICKLE_FILENAME,
        OUTPUT_LIST,
        FAISS_INDEX_FILENAME,
        MODEL_PATH
    )
