import pickle
import numpy as np
import spacy
import faiss
from tqdm import tqdm
import warnings
import gc


warnings.simplefilter("ignore")


def create_quantized_index(embeddings_np, d):
    data_size = len(embeddings_np)

    # More conservative `nlist` calculation:
    nlist = max(1, int(data_size ** 0.5))  # Use cube root for smaller values

    # Ensure `m` is a divisor of `d` (300), and adjust dynamically
    possible_m_values = [i for i in range(1, min(30, d) + 1) if d % i == 0]
    m = max(4, min(possible_m_values, key=lambda x: abs(x - (data_size // 100))))

    # Create the quantizer and index
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    # Train the index if there are enough data points
    if data_size >= nlist * 39:
        index.train(embeddings_np)
    else:
        print(
            f"Warning: Insufficient data points ({data_size}) for {nlist} centroids. Consider reducing `nlist` or adding more data.")
        return None  # Return None if training fails

    return index

def create_flat_index(embeddings_np, d):
    """Create a flat index for smaller datasets."""
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    return index


# calculate mean vector from the model's vocabulary
def calculate_mean_vector(model):
    vectors = [word.vector for word in model.vocab if word.has_vector]
    if vectors:
        mean_vector = np.mean(vectors, axis=0)
        return mean_vector
    else:
        return np.zeros((300,))


# get average embeddings for terms in batches using the user's model, with mean vector fallback
def get_average_embeddings_batched(terms, model, mean_vector):
    docs = list(model.pipe(terms))
    embeddings = []
    for doc in docs:
        valid_vectors = [token.vector for token in doc if token.has_vector and token.vector.shape[0] == 300]
        if len(valid_vectors) == 0:
            embeddings.append(mean_vector)
        else:
            average_embedding = np.mean(valid_vectors, axis=0)
            embeddings.append(average_embedding)
    return embeddings

# Preprocessing and indexing function to create Faiss index and save necessary data
def preprocess_and_index(term_id_dict, output_pickle_filename, output_list, faiss_index_filename, model_path, batch_size=10000):
    # Load your custom embedding model
    nlp_model = spacy.load(model_path)
    mean_vector = calculate_mean_vector(nlp_model)

    embeddings = []
    indexed_terms = []

    terms = list(term_id_dict.keys())
    ids = list(term_id_dict.values())

    for idx in tqdm(range(0, len(terms), batch_size), desc="Generating Embeddings"):
        term_batch = terms[idx: idx + batch_size]
        id_batch = ids[idx: idx + batch_size]

        batch_embeddings = get_average_embeddings_batched(term_batch, nlp_model, mean_vector)

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
    # Choose the index type based on the size of embeddings
    if len(embeddings_np) < 10000:
        print("Dataset is small; using flat index...")
        index = create_flat_index(embeddings_np, d)
    else:
        index = create_quantized_index(embeddings_np, d)
        if index is None:
            print("Failed to create a trained index. Exiting.")
            return  # Stop if training fails

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
