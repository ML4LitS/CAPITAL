import pickle
import numpy as np
import spacy
import faiss
from tqdm import tqdm
import warnings
import re
import gc

warnings.simplefilter("ignore")


def create_quantized_index(embeddings_np, d):
    nlist = 1000
    m = 30
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    index.train(embeddings_np)
    return index


def get_average_embeddings_batched(terms, model):
    docs = list(model.pipe(terms))
    embeddings = []
    for doc in docs:
        valid_vectors = [token.vector for token in doc if token.has_vector and token.vector.shape[0] == 300]
        if len(valid_vectors) == 0:
            embeddings.append(np.zeros((300,)))
        else:
            average_embedding = np.mean(valid_vectors, axis=0)
            embeddings.append(average_embedding)
    return embeddings


def preprocess_and_index(term_id_dict, output_pickle_filename, output_list, faiss_index_filename, model_path, batch_size=10000):
    nlp_model = spacy.load(model_path)

    embeddings = []
    indexed_terms = []

    terms = list(term_id_dict.keys())
    ids = list(term_id_dict.values())

    for idx in tqdm(range(0, len(terms), batch_size), desc="Generating Embeddings"):
        term_batch = terms[idx: idx + batch_size]
        id_batch = ids[idx: idx + batch_size]

        batch_embeddings = get_average_embeddings_batched(term_batch, nlp_model)

        for term, term_id, embedding in zip(term_batch, id_batch, batch_embeddings):
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print(f"Term '{term}' with ID '{term_id}' has a zero vector.")
            normalized_embedding = embedding if norm == 0 else embedding / norm
            embeddings.append(normalized_embedding)
            indexed_terms.append(term)
        gc.collect()

    d = 300
    embeddings_np = np.array(embeddings).astype('float32')
    index = create_quantized_index(embeddings_np, d)
    index.add(embeddings_np)
    del embeddings, embeddings_np
    gc.collect()

    print("Saving quantized faiss index...")
    faiss.write_index(index, faiss_index_filename)

    print("Saving term to ID mapping and indexed terms...")
    with open(output_pickle_filename, "wb") as outfile:
        pickle.dump({"term_to_id": term_id_dict, "indexed_terms": indexed_terms}, outfile)

    print("Writing terms to a txt file...")
    with open(output_list, "w") as txt_file:
        for term in term_id_dict.keys():
            txt_file.write(term + "\n")

# You can then import indexing.py and use preprocess_and_index function by passing the dictionary of terms and IDs.
