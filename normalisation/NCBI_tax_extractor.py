# import csv
# from tqdm import tqdm
# import re
# import pickle
# import json
#
# # Input and output filenames
# input_filename = "/home/stirunag/work/github/source_data/knowledge_base/taxdump/names.dmp"
# output_pickle_filename = "/home/stirunag/work/github/source_data/dictionaries/NCBI_taxonomy.pkl"
# output_jsonl_filename = "/home/stirunag/work/github/source_data/training_data/train_data_floret.jsonl"
#
# # Determine the total number of rows in the input file for the progress bar
# with open(input_filename, 'r') as f:
#     total_rows = sum(1 for line in f)
#
# # Dictionary to hold the output data
# output_dict = {}
#
#
# # Function to process the content of column 1
# def process_column_content(s):
#     s = re.sub(r'\(.*?\)|\".*?\"|\[.*?\]', '', s).strip()
#     return s.strip()
#
#
# # Read the .dmp file and process the data
# with open(input_filename, "r") as infile:
#     # Create CSV reader object
#     reader = csv.reader(infile, delimiter="|")
#
#     # Iterate through each row in the input file with a progress bar
#     for row in tqdm(reader, total=total_rows, desc="Processing"):
#         if "authority" not in row[3]:
#             extracted_text = process_column_content(row[1])
#             # Update the dictionary with the extracted text and corresponding identifier
#             output_dict[extracted_text] = row[0].strip()
#
# # Dump the dictionary as a pickle file
# with open(output_pickle_filename, "wb") as outfile:
#     pickle.dump(output_dict, outfile)
#
# # Append data to jsonl file
# with open(output_jsonl_filename, "a") as jsonl_file:
#     for term in output_dict.keys():
#         json_line = json.dumps({"text": term})
#         jsonl_file.write(json_line + "\n")
#



# # Filenames
# input_filename = "/home/stirunag/work/github/source_data/knowledge_base/taxdump/names.dmp"
# output_pickle_filename = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/NCBI_terms.pkl"
# output_list = "/home/stirunag/work/github/source_data/training_data/NCBI_list.txt"
# faiss_index_filename = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/NCBI_terms.index"


import csv
import pickle
import numpy as np
import spacy
import faiss
from tqdm import tqdm
import warnings
import re
import gc

# Constants
INPUT_FILENAME = "/home/stirunag/work/github/source_data/knowledge_base/taxdump/names.dmp"
OUTPUT_PICKLE_FILENAME ="/home/stirunag/work/github/ML_annotations/normalisation/dictionary/NCBI_terms.pkl"
OUTPUT_LIST = "/home/stirunag/work/github/source_data/training_data/NCBI_list.txt"
FAISS_INDEX_FILENAME = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/NCBI_terms.index"
OUTPUT_INDEXED_TERMS_FILENAME = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/NCBI_indexed_terms.pkl"
# Turn off warnings
warnings.simplefilter("ignore")

# Load the spaCy model
nlp = spacy.load("/home/stirunag/work/github/ML_annotations/normalisation/en_floret_model")

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
        valid_vectors = [token.vector for token in doc if token.has_vector and token.vector.shape[0] == 300]

        if len(valid_vectors) == 0:
            embeddings.append(np.zeros((300,)))
        else:
            average_embedding = np.mean(valid_vectors, axis=0)
            embeddings.append(average_embedding)

    return embeddings

def process_column_content(s):
    """Clean and strip unwanted characters."""
    return re.sub(r'\(.*?\)|\".*?\"|\[.*?\]', '', s).strip().lower()

try:
    print("Loading ontology...")

    term_to_id = {}
    embeddings = []
    indexed_terms = []

    with open(INPUT_FILENAME, 'r') as f:
        total_rows = sum(1 for line in f)

    BATCH_SIZE = 10000
    term_batches = []
    id_batches = []
    current_batch_terms = []
    current_batch_ids = []

    with open(INPUT_FILENAME, "r") as infile:
        reader = csv.reader(infile, delimiter="|")
        for row in tqdm(reader, total=total_rows, desc="Reading CSV"):
            if "authority" not in row[3]:
                term_name = process_column_content(row[1])

                # Check for empty or single character terms and skip them
                if not term_name or len(term_name) <= 1:
                    continue

                term_id = row[0].strip()

                current_batch_terms.append(term_name)
                current_batch_ids.append(term_id)

                if len(current_batch_terms) == BATCH_SIZE:
                    term_batches.append(current_batch_terms)
                    id_batches.append(current_batch_ids)
                    current_batch_terms = []
                    current_batch_ids = []

    if current_batch_terms:
        term_batches.append(current_batch_terms)
        id_batches.append(current_batch_ids)

    for term_batch, id_batch in tqdm(zip(term_batches, id_batches), total=len(term_batches),
                                     desc="Generating Embeddings"):
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
        gc.collect()

    d = 300
    embeddings_np = np.array(embeddings).astype('float32')
    index = create_quantized_index(embeddings_np, d)
    index.add(embeddings_np)

    # Free up memory after using embeddings_np
    del embeddings, embeddings_np
    gc.collect()

    print("Saving quantized faiss index...")
    faiss.write_index(index, FAISS_INDEX_FILENAME)

    # print("Saving term to ID mapping...")
    # with open(OUTPUT_PICKLE_FILENAME, "wb") as outfile:
    #     pickle.dump(term_to_id, outfile)

    print("Saving term to ID mapping and indexed terms...")
    with open(OUTPUT_PICKLE_FILENAME, "wb") as outfile:
        pickle.dump({"term_to_id": term_to_id, "indexed_terms": indexed_terms}, outfile)


    print("Writing terms to a txt file...")
    with open(OUTPUT_LIST, "w") as txt_file:
        for term in term_to_id.keys():
            txt_file.write(term + "\n")

    # print("Saving indexed terms list...")
    # with open(OUTPUT_INDEXED_TERMS_FILENAME, "wb") as outfile:
    #     pickle.dump(indexed_terms, outfile)
    # print("Done!")

except Exception as e:
    print(f"An error occurred: {e}")

