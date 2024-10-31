from bs4 import BeautifulSoup
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
    # Parse the OWL file using BeautifulSoup
    with open(input_filename, 'r') as file:
        soup = BeautifulSoup(file, 'xml')

    term_to_id = {}

    # Find all owl:Class elements
    all_classes = soup.find_all("owl:Class")

    # Iterate through each class and extract relevant information
    for owl_class in tqdm(all_classes, desc="Extracting ChEBI Terms", total=len(all_classes)):
        # Extract the ID from rdf:about attribute
        chebi_id = owl_class.get('rdf:about')
        if chebi_id:
            chebi_id = chebi_id.split("/")[-1]

        # Initialize term_name to None
        term_name = None

        # Extract the label using rdfs:label
        label = owl_class.find("rdfs:label")
        if label:

            term_name = label.text.replace('atom','').replace('molecular entity','').strip()

        # Make sure we have a term name and ID
        if term_name and chebi_id:

            term_to_id[term_name.lower()] = chebi_id

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
