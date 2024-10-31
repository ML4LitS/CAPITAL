import csv
import gzip
from tqdm import tqdm
from indexing import preprocess_and_index
import string
import re

resources = [
    'HL7V2.5', 'ICD10AM', 'ICD10AMAE', 'LCH', 'MTHICPC2ICD10AE', 'MTHMST',
    'NCI_RENI', 'SNM', 'SNMI', 'SNOMEDCT_VET', 'FMA', 'GO', 'ICD10', 'ICD10AE',
    'ICD10CM', 'ICD9CM', 'LNC', 'MDR', 'MEDLINEPLUS', 'MSH', 'MTH', 'MTHICD9',
    'NCI', 'NCI_BRIDG', 'NCI_CDISC', 'NCI_CTCAE', 'NCI_CTEP-SDC', 'NCI_FDA',
    'NCI_NCI-GLOSS', 'NDFRT', 'OMIM', 'SNOMEDCT_US', 'WHO'
]


import re
import string

# List of phrases to remove (converted to lowercase)
phrases_to_remove = [
    '--', 'physical finding', 'diagnosis', 'disorder', 'procedure', 'finding',
    'symptom', 'history', 'treatment', 'manifestation', 'disease', 'morphologic abnormality',
    'etiology', 'observable entity', 'event', 'situation', 'degrees', 'in some patients', 'cm',
    'mm', '#', 'rare', 'degree', 'including anastomotic', 'navigational concept',
    '1 patient', 'qualifier value', 'lab test', 'unintentional', 'tophi', 'nos',
    'msec', 'reni', 'less common', 'symptom'
]

# Replacement patterns (e.g., phrases or patterns to replace or remove)
replacement_patterns = [
    r'-- ', r' \(physical finding\)', r' \(diagnosis\)', r' \(disorder\)', r' \(procedure\)',
    r' \(finding\)', r' \(symptom\)', r' \(history\)', r' \(treatment\)', r' \(manifestation\)',
    r' \[Disease/Finding\]', r' \(morphologic abnormality\)', r' \(etiology\)', r' \(observable entity\)',
    r' \(event\)', r' \(situation\)', r' \(___ degrees\)', r' \(in some patients\)',
    r' \(___ cm\)', r' \(___ mm\)', r' \(#___\)', r' \(rare\)', r' \(___ degree\.\)',
    r' \(including anastomotic\)', r' \(navigational concept\)', r' \(___cm\)', r' \(1 patient\)',
    r' \(qualifier value\)', r' \(lab test\)', r' \(unintentional\)', r' \(tophi\)', r' \(NOS\)',
    r' \(___ msec\)', r' \(RENI\)', r' \(less common\)', r' \[as symptom\]', r' \(s\)'
]

def clean_and_modify_term(term):
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
    term = ' '.join(term.split())

    return term.lower()


def is_required_category(category):
    required_categories = ["T020", "T190", "T049", "T019", "T047", "T050", "T033", "T037", "T048", "T191", "T046", "T184"]
    return category in required_categories

def extract_terms_and_ids_from_umls(input_files, mrsty_file):
    """
    Extract terms and IDs from the provided UMLS into a dictionary.

    Args:
        input_files (list): List of paths to the UMLS files.
        mrsty_file (str): Path to the MRSTY file.

    Returns:
        dict: Dictionary where keys are terms and values are IDs.
    """

    term_to_id = {}

    print("Processing terms from UMLS..")

    # Read each row in MRSTY file and check for the required category
    interested_ids = set()
    with gzip.open(mrsty_file, 'rt') as file:
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            if is_required_category(row[1]):
                interested_ids.add(row[0])

    # Read each row in input_files
    for filename in input_files:
        with gzip.open(filename, 'rt') as file:
            reader = csv.reader(file, delimiter='|')
            for row in tqdm(reader, desc=f"Processing {filename}"):
                if len(row) > 16:  # Ensure there are enough columns in the row
                    if (row and len(row[14]) > 3 and row[1] == "ENG" and row[16] != "0" and
                            row[11] in resources and row[0] in interested_ids):
                        term = clean_and_modify_term(row[14])
                        term_to_id[term] = row[0]

    return term_to_id

if __name__ == "__main__":
    path_ = "/home/stirunag/work/github/source_data/knowledge_base/umls-2022AB-full/"
    input_files = [path_ + 'MRCONSO.RRF.aa.gz', path_ + 'MRCONSO.RRF.ab.gz']
    mrsty_file = path_ + "MRSTY.RRF.gz"

    term_id_dict = extract_terms_and_ids_from_umls(input_files, mrsty_file)

    OUTPUT_PICKLE_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/umls_terms.pkl"
    OUTPUT_LIST = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/umls_list.txt"
    FAISS_INDEX_FILENAME = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/umls_terms.index"
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
