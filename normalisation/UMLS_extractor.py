import gzip
import csv
import pickle
import numpy as np
import spacy
import faiss
from tqdm import tqdm
import warnings

warnings.simplefilter("ignore")

# Load the spaCy model
nlp = spacy.load("/home/stirunag/work/github/ML_annotations/normalisation/en_floret_model")

resources = [
    'HL7V2.5',
    'ICD10AM',
    'ICD10AMAE',
    'LCH',
    'MTHICPC2ICD10AE',
    'MTHMST',
    'NCI_RENI',
    'SNM',
    'SNMI',
    'SNOMEDCT_VET',
    'FMA',
    'GO',
    'ICD10',
    'ICD10AE',
    'ICD10CM',
    'ICD9CM',
    'LNC',
    'MDR',
    'MEDLINEPLUS',
    'MSH',
    'MTH',
    'MTHICD9',
    'NCI',
    'NCI_BRIDG',
    'NCI_CDISC',
    'NCI_CTCAE',
    'NCI_CTEP-SDC',
    'NCI_FDA',
    'NCI_NCI-GLOSS',
    'NDFRT',
    'OMIM',
    'SNOMEDCT_US',
    'WHO'
]


def filter_term(term):
    if len(term) < 3:
        return False
    return True


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

    return term


def is_required_category(category):
    required_categories = ["T020", "T190", "T049", "T019", "T047", "T050", "T033", "T037", "T048", "T191", "T046",
                           "T184"]
    return category in required_categories


# Define the list of input files

path_ = "/home/stirunag/work/github/source_data/knowledge_base/umls-2022AB-full/"
input_files = [path_ + 'MRCONSO.RRF.aa.gz', path_ + 'MRCONSO.RRF.ab.gz']
mrsty_file = path_ + "MRSTY.RRF.gz"
output_pickle_filename = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/umls_terms.pkl"
output_list = "/home/stirunag/work/github/source_data/training_data/umls_list.txt"
faiss_index_filename = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/umls_terms.index"



def create_quantized_index(embeddings_np, d):
    """Create a trained IVFPQ index."""
    nlist = 1000
    m = 30
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    index.train(embeddings_np)
    return index

def get_average_embeddings_batched(terms):
    """Return average embeddings for terms."""
    docs = list(nlp.pipe(terms))
    embeddings = []

    for doc in docs:
        valid_vectors = [token.vector for token in doc if token.has_vector and token.vector_norm != 0 and token.vector.shape[0] == 300]

        if len(valid_vectors) == 0:
            embeddings.append(np.zeros((300,)))
        else:
            average_embedding = np.mean(valid_vectors, axis=0)
            embeddings.append(average_embedding)

    return embeddings


# Read each row in MRSTY file and check for the required category
interested_ids = set()
with gzip.open(mrsty_file, 'rt') as file:
    reader = csv.reader(file, delimiter='|')
    for row in reader:
        if is_required_category(row[1]):
            interested_ids.add(row[0])

# Read each row in input_files
terms_dict = {}
for filename in input_files:
    with gzip.open(filename, 'rt') as file:
        reader = csv.reader(file, delimiter='|')
        for row in tqdm(reader, desc=f"Processing {filename}"):
            if len(row) > 16:  # Ensure there are enough columns in the row
                if (row and len(row[14]) > 3 and row[1] == "ENG" and row[16] != "0" and
                        row[11] in resources and row[0] in interested_ids):
                    term = modify_term(row[14])
                    terms_dict[term] = row[0]



print("Processing terms and generating embeddings...")

term_to_id = {}
embeddings = []
indexed_terms = []

BATCH_SIZE = 100
term_batches = []
id_batches = []
current_batch_terms = []
current_batch_ids = []

for term, umls_id in tqdm(terms_dict.items(), desc="Extracting Terms"):
    if not filter_term(term):  # Filter the terms using your filter function
        continue

    current_batch_terms.append(term)
    current_batch_ids.append(umls_id)

    if len(current_batch_terms) == BATCH_SIZE:
        term_batches.append(current_batch_terms)
        id_batches.append(current_batch_ids)
        current_batch_terms = []
        current_batch_ids = []

if current_batch_terms:
    term_batches.append(current_batch_terms)
    id_batches.append(current_batch_ids)

for term_batch, id_batch in tqdm(zip(term_batches, id_batches), total=len(term_batches), desc="Generating Embeddings"):
    batch_embeddings = get_average_embeddings_batched(term_batch)

    for term, term_id, embedding in zip(term_batch, id_batch, batch_embeddings):
        norm = np.linalg.norm(embedding)

        # Check if the embedding is a zero vector
        if norm == 0:
            print(f"Term '{term}' with ID '{term_id}' has a zero vector.")

        # Normalizing the vector
        normalized_embedding = embedding if norm == 0 else embedding / norm
        embeddings.append(normalized_embedding)
        term_to_id[term] = term_id
        indexed_terms.append(term)

    # Clear out the current batch to free up memory
    del term_batch, id_batch, batch_embeddings

d = 300
embeddings_np = np.array(embeddings).astype('float32')
index = create_quantized_index(embeddings_np, d)
index.add(embeddings_np)

print("Saving quantized faiss index...")
faiss.write_index(index, faiss_index_filename)

print("Saving term to ID mapping and indexed terms...")
with open(output_pickle_filename, "wb") as outfile:
    pickle.dump({"term_to_id": term_to_id, "indexed_terms": indexed_terms}, outfile)

print("Writing terms to a txt file...")
with open(output_list, "w") as txt_file:
    for term in term_to_id.keys():
        txt_file.write(term + "\n")

print("Done!")




# # Write output csv
# with open(output_file, 'w') as outfile:
#     writer = csv.writer(outfile)
#     for term, val in terms_dict.items():
#         writer.writerow([val, term])
#
# # Pickle dump the dictionary
# with open(dict_output_file, 'wb') as file:
#     pickle.dump(terms_dict, file)
#
# print(f"Output written to {output_file}")
# print(f"Dictionary pickled to {dict_output_file}")
#
# print("Processing complete!")
