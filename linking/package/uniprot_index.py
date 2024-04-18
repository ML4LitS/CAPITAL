import re
from tqdm import tqdm
from indexing import preprocess_and_index

# Patterns for extracting relevant information
AC_PATTERN = re.compile(r"AC   (.*?);\n")
GN_PATTERN = re.compile(r"GN   (.+?)\n")
DE_PATTERN = re.compile(r"DE   (.+?)\n")

GN_KEYS = ['Name', 'Synonyms', 'OrderedLocusNames', 'ORFNames', 'EC']
DE_KEYS = ['RecName: Full', 'AltName: Full', 'Short']


def extract_values_from_line(line, keys):
    values_list = []
    line = re.sub(r"\{.*?\}", "", line).strip()
    for key in keys:
        match = re.search(f"{key}=(.*?)(;|$)", line)
        if match:
            values = match.group(1).split(', ')
            values_list.extend(values)
    return values_list


def process_document(buffer, output_dict):
    doc = ''.join(buffer)

    ac_match = AC_PATTERN.search(doc)
    ac_values = ac_match.group(1).split("; ") if ac_match else None

    if ac_values:
        ac_value = ac_values[0]  # using the primary AC value
        de_matches = DE_PATTERN.findall(doc)
        for de_line in de_matches:
            de_values = extract_values_from_line(de_line, DE_KEYS)
            for value in de_values:
                output_dict[value.strip()] = ac_value

        gn_matches = GN_PATTERN.findall(doc)
        for gn_line in gn_matches:
            gn_values = extract_values_from_line(gn_line, GN_KEYS)
            for value in gn_values:
                output_dict[value.strip()] = ac_value


def extract_terms_and_ids_from_uniprot(input_filename):
    """
    Extract terms and IDs from the provided UniProt file into a dictionary.

    Args:
        input_filename (str): Path to the UniProt file.

    Returns:
        dict: Dictionary where keys are terms and values are IDs.
    """
    buffer = []
    term_to_id = {}

    print("Processing terms from UniProt...")

    with open(input_filename, 'r') as file:
        for line in tqdm(file, desc="Processing file"):
            buffer.append(line)
            if line.startswith("//"):
                process_document(buffer, term_to_id)
                buffer = []

    return term_to_id


if __name__ == "__main__":
    input_filename = "/home/stirunag/work/github/source_data/knowledge_base/uniprot/uniprot_sprot.dat"
    term_id_dict = extract_terms_and_ids_from_uniprot(input_filename)

    # After this, you can use term_id_dict in other functions for further processing.
    OUTPUT_PICKLE_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/uniprot_terms.pkl"
    OUTPUT_LIST = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/uniprot_list.txt"
    FAISS_INDEX_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/uniprot_terms.index"
    MODEL_PATH = "/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model"

    # After this, you can use term_id_dict in other functions for further processing.
    # Preprocess and Index the terms
    preprocess_and_index(
        term_id_dict,
        OUTPUT_PICKLE_FILENAME,
        OUTPUT_LIST,
        FAISS_INDEX_FILENAME,
        MODEL_PATH
    )