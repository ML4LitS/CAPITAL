import csv
import gzip
from tqdm import tqdm
from indexing import preprocess_and_index


resources = [
    'HL7V2.5', 'ICD10AM', 'ICD10AMAE', 'LCH', 'MTHICPC2ICD10AE', 'MTHMST',
    'NCI_RENI', 'SNM', 'SNMI', 'SNOMEDCT_VET', 'FMA', 'GO', 'ICD10', 'ICD10AE',
    'ICD10CM', 'ICD9CM', 'LNC', 'MDR', 'MEDLINEPLUS', 'MSH', 'MTH', 'MTHICD9',
    'NCI', 'NCI_BRIDG', 'NCI_CDISC', 'NCI_CTCAE', 'NCI_CTEP-SDC', 'NCI_FDA',
    'NCI_NCI-GLOSS', 'NDFRT', 'OMIM', 'SNOMEDCT_US', 'WHO'
]

def modify_term(term):
    replacements = [
        '-- ',
        ' (physical finding)', ' (diagnosis)', ' (disorder)', ' (procedure)', ' (finding)',
        ' (symptom)', ' (history)', ' (treatment)', ' (manifestation)', ' [Disease/Finding]',
        ' (morphologic abnormality)', ' (etiology)', ' (observable entity)', ' (event)',
        ' (situation)', ' (___ degrees)', ' (in some patients)', ' (___ cm)', ' (___ mm)',
        ' (#___)', ' (rare)', ' (___ degree.)', ' (including anastomotic)', ' (navigational concept)',
        ' (___cm)', ' (1 patient)', ' (qualifier value)', ' (lab test)', ' (unintentional)',
        ' (tophi)', ' (NOS)', ' (___ msec)', ' (RENI)', ' (less common)', ' [as symptom]', ' (s)'
    ]
    for replacement in replacements:
        term = term.replace(replacement, '')
    term = term.replace('-', ' ')
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
                        term = modify_term(row[14])
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
