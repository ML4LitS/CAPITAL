import re
import json
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

input_filename = "/home/stirunag/work/github/source_data/knowledge_base/uniprot/uniprot_sprot.dat"
output_pickle_filename = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/uniprot_terms.pkl"
output_list = "/home/stirunag/work/github/source_data/training_data/uniprot_list.txt"
faiss_index_filename = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/uniprot_terms.index"

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

def process_file_line_by_line(filename):
    buffer = []
    output_dict = {}

    with open(filename, 'r') as file:
        for line in tqdm(file, desc="Processing file"):
            buffer.append(line)
            if line.startswith("//"):
                process_document(buffer, output_dict)
                buffer = []

    return output_dict

term_to_id = process_file_line_by_line(input_filename)

BATCH_SIZE = 1000
embeddings = []
indexed_terms = []

terms = list(term_to_id.keys())
for i in tqdm(range(0, len(terms), BATCH_SIZE), desc="Generating Embeddings"):
    term_batch = terms[i: i+BATCH_SIZE]
    batch_embeddings = get_average_embeddings_batched(term_batch)

    for term, embedding in zip(term_batch, batch_embeddings):
        norm = np.linalg.norm(embedding)
        normalized_embedding = embedding if norm == 0 else embedding / norm
        embeddings.append(normalized_embedding)
        indexed_terms.append(term)

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








# import re
# import pickle
# import csv
# import json
# from tqdm import tqdm
#
# # Patterns for extracting relevant information
# AC_PATTERN = re.compile(r"AC   (.*?);\n")
# GN_PATTERN = re.compile(r"GN   (.+?)\n")
# DE_PATTERN = re.compile(r"DE   (.+?)\n")
#
# # List of keys we are interested in, for GN and DE values
# GN_KEYS = ['Name', 'Synonyms', 'OrderedLocusNames', 'ORFNames', 'EC']
# DE_KEYS = ['RecName: Full', 'AltName: Full', 'Short']
#
# output_pickle_filename = "/home/stirunag/work/github/source_data/dictionaries/output.pkl"
# output_csv_filename = "/home/stirunag/work/github/source_data/dictionaries/output.csv"
# output_jsonl_filename = "/home/stirunag/work/github/source_data/training_data/train_data_floret.jsonl"
#
# def extract_values_from_line(line, keys):
#     values_list = []
#     # Remove text enclosed in { }
#     line = re.sub(r"\{.*?\}", "", line).strip()
#     for key in keys:
#         match = re.search(f"{key}=(.*?)(;|$)", line)
#         if match:
#             values = match.group(1).split(', ')
#             values_list.extend(values)
#     return values_list
#
#
# def process_document(buffer, output_dict):
#     doc = ''.join(buffer)
#
#     ac_match = AC_PATTERN.search(doc)
#     ac_values = ac_match.group(1).split("; ") if ac_match else None
#
#     if ac_values:
#         ac_value = ac_values[0]  # using the primary AC value
#         # For DE values
#         de_matches = DE_PATTERN.findall(doc)
#         for de_line in de_matches:
#             de_values = extract_values_from_line(de_line, DE_KEYS)
#             for value in de_values:
#                 output_dict[value.strip()] = ac_value  # interchanging keys and values
#
#         # For GN values
#         gn_matches = GN_PATTERN.findall(doc)
#         for gn_line in gn_matches:
#             gn_values = extract_values_from_line(gn_line, GN_KEYS)
#             for value in gn_values:
#                 output_dict[value.strip()] = ac_value  # interchanging keys and values
#
#
# def process_file_line_by_line(filename):
#     buffer = []
#     output_dict = {}
#
#     with open(filename, 'r') as file:
#         for line in tqdm(file, desc="Processing file"):
#             buffer.append(line)
#             if line.startswith("//"):
#                 process_document(buffer, output_dict)
#                 buffer = []
#
#     return output_dict
#
#
# filename = "/home/stirunag/work/github/source_data/knowledge_base/uniprot/uniprot_sprot.dat"
# output_dict = process_file_line_by_line(filename)
#
# # Dump the dictionary as a pickle file
# with open(output_pickle_filename, "wb") as outfile:
#     pickle.dump(output_dict, outfile)
#
# # Write to CSV file
# with open(output_csv_filename, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Term', 'AC'])  # header row
#     for key, value in output_dict.items():
#         writer.writerow([key, value])
#
# # Append data to jsonl file
# with open(output_jsonl_filename, "a") as jsonl_file:
#     for key, value in output_dict.items():
#         json_line = json.dumps({"text": key, "AC": value})  # getting term and AC for jsonl
#         jsonl_file.write(json_line + "\n")