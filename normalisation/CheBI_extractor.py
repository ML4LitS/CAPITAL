import csv
import pickle
import numpy as np
from pronto import Ontology
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

#
def get_average_embeddings_batched(terms):
    """Return average embeddings for terms."""
    docs = list(nlp.pipe(terms))
    embeddings = []

    for doc in docs:
        # Filtering out tokens without vectors or with unexpected vector sizes
        valid_vectors = [token.vector for token in doc if token.has_vector and token.vector_norm != 0 and token.vector.shape[0] == 300]

        # If no valid vectors, append a zero vector
        if len(valid_vectors) == 0:
            embeddings.append(np.zeros((300,)))
        else:
            average_embedding = np.mean(valid_vectors, axis=0)
            embeddings.append(average_embedding)

    return embeddings

# Filenames
input_filename = "/home/stirunag/work/github/source_data/knowledge_base/chebi/chebi.owl"
output_pickle_filename = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/chebi_terms.pkl"
output_list = "/home/stirunag/work/github/source_data/training_data/chebi_list.txt"
faiss_index_filename = "/home/stirunag/work/github/ML_annotations/normalisation/dictionary/chebi_terms.index"

print("Loading ontology...")
chebi = Ontology(input_filename)

term_to_id = {}
embeddings = []
indexed_terms = []

BATCH_SIZE = 100
term_batches = []
id_batches = []
current_batch_terms = []
current_batch_ids = []

print("Processing terms and generating embeddings...")
for term in tqdm(chebi.terms(), total=len(chebi.terms()), desc="Extracting Terms"):
    chebi_id = term.id
    term_name = term.name

    if not term_name:
        continue

    term_name = term_name.lower()

    current_batch_terms.append(term_name)
    current_batch_ids.append(chebi_id)

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

