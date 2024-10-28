from bs4 import BeautifulSoup
from indexing import preprocess_and_index

def extract_terms_and_ids_from_mwt(input_filename):
    """
    Extract terms and IDs from the provided MWT file into a dictionary.

    Args:
        input_filename (str): Path to the MWT file.

    Returns:
        dict: Dictionary where keys are terms and values are IDs.
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

        if term_name and len(term_name) > 1:
            term_to_id[term_name] = term_id

    return term_to_id



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