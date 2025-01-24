import faiss
import pickle
import spacy
import sys
import logging
import re
from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
import string
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load paths from .env_paths
MODEL_PATH = os.getenv("SPACY_MODEL_PATH")
BASE_PATH = os.getenv("BASE_DICTIONARY_PATH")


class EntityLinker:
    """
    A class to handle loading annotation data and performing entity linking.
    """

    def __init__(self, env_path='/home/stirunag/work/github/CAPITAL/daily_pipeline/.env_paths'):
        """
        Initialize the EntityLinker by loading environment variables,
        the spaCy model, and annotations.

        Args:
            env_path (str): Path to the .env_paths file.
        """
        # Load environment variables
        load_dotenv(env_path)
        self.MODEL_PATH = os.getenv("SPACY_MODEL_PATH")
        self.BASE_PATH = os.getenv("BASE_DICTIONARY_PATH")

        # Verify paths are loaded correctly
        if not self.MODEL_PATH or not self.BASE_PATH:
            raise ValueError("SPACY_MODEL_PATH and BASE_DICTIONARY_PATH must be set in .env_paths")

        # Load spaCy model
        try:
            logging.info("Loading spaCy model for entity linking...")
            self.nlp = spacy.load(self.MODEL_PATH)
            logging.info("SpaCy model loaded successfully.")
        except OSError:
            logging.error(f"The spaCy model at '{self.MODEL_PATH}' was not found.")
            sys.exit(1)

        # Mapping of annotation types to their respective index and pickle files
        self.FILE_MAPPING = {
            'CD': ('chebi_terms.index', 'chebi_terms.pkl'),
            'OG': ('NCBI_terms.index', 'NCBI_terms.pkl'),
            'DS': ('umls_terms.index', 'umls_terms.pkl'),
            'GP': ('uniprot_terms.index', 'uniprot_terms.pkl'),
            'GO': ('go_terms.index', 'go_terms.pkl'),
            'EM': ('em_terms.index', 'em_terms.pkl'),
            'primer': ('primer_terms.index', 'primer_terms.pkl'),  # Ensure 'Primer' is correctly capitalized
            'EFO': ('EFO_terms.index', 'EFO_terms.pkl'),
            'ENVO': ('ENVO_terms.index', 'ENVO_terms.pkl')
        }

        # Define phrases to remove
        self.phrases_to_remove = [
            '--', 'physical finding', 'diagnosis', 'disorder', 'procedure', 'finding',
            'symptom', 'history', 'treatment', 'manifestation', 'disease', 'finding',
            'morphologic abnormality', 'etiology', 'observable entity', 'event',
            'situation', 'degrees', 'in some patients', 'cm', 'mm',
            '#', 'rare', 'degree', 'including anastomotic', 'navigational concept',
            '1 patient', 'qualifier value', 'lab test', 'unintentional',
            'tophi', 'nos', 'msec', 'reni', 'less common', 'as symptom'
        ]

        # Initialize loaded_data
        self.loaded_data = {}

    def load_annotations(self, annotation_types=None):
        """
        Load specified annotation types or all if none are specified.

        Args:
            annotation_types (list or None): List of annotation type keys to load.
                                             If None, all annotation types are loaded.

        Returns:
            dict: A dictionary containing loaded data for each annotation type.
                  The structure is:
                  {
                      'AnnotationType': {
                          'term_to_id': {...},
                          'indexed_terms': [...],
                          'index': faiss.Index,
                          'id_to_term': {...}
                      },
                      ...
                  }
        """
        # If no specific annotation types are provided, load all
        if annotation_types is None:
            annotation_types = list(self.FILE_MAPPING.keys())
            logging.info("No specific annotation types provided. Loading all annotation types.")
        else:
            # Validate provided annotation types
            invalid_types = [atype for atype in annotation_types if atype not in self.FILE_MAPPING]
            if invalid_types:
                logging.error(f"The following annotation types are invalid: {', '.join(invalid_types)}")
                logging.error(f"Valid annotation types are: {', '.join(self.FILE_MAPPING.keys())}")
                sys.exit(1)
            logging.info(f"Loading specified annotation types: {', '.join(annotation_types)}")

        for annotation_type in annotation_types:
            index_file, pkl_file = self.FILE_MAPPING[annotation_type]
            index_path = os.path.join(self.BASE_PATH, index_file)
            pkl_path = os.path.join(self.BASE_PATH, pkl_file)

            # Load pickle file
            try:
                with open(pkl_path, "rb") as infile:
                    data = pickle.load(infile)
                logging.info(f"Loaded pickle file '{pkl_file}' for annotation type '{annotation_type}'.")
            except FileNotFoundError:
                logging.error(f"Pickle file '{pkl_file}' for annotation type '{annotation_type}' not found.")
                continue
            except pickle.UnpicklingError:
                logging.error(f"Failed to unpickle file '{pkl_file}' for annotation type '{annotation_type}'.")
                continue

            # Load FAISS index
            try:
                index = faiss.read_index(index_path)
                logging.info(f"Loaded FAISS index '{index_file}' for annotation type '{annotation_type}'.")
            except Exception as e:
                logging.error(f"Failed to load FAISS index '{index_file}' for annotation type '{annotation_type}'. Exception: {e}")
                continue

            # Create a reverse mapping from term_id to term
            id_to_term = {v: k for k, v in data["term_to_id"].items()}

            # Store loaded data
            self.loaded_data[annotation_type] = {
                "term_to_id": data["term_to_id"],
                "indexed_terms": data.get("indexed_terms", []),  # Assuming it's a list
                "index": index,
                "id_to_term": id_to_term
            }

            logging.info(f"Loaded data for annotation type '{annotation_type}'.")

        return self.loaded_data

    def clean_term(self, term):
        """
        Clean the term by converting it to lowercase, removing specified phrases,
        punctuation, and extra whitespace.

        Args:
            term (str): The term to clean.

        Returns:
            str: The cleaned term.
        """
        # Convert term to lowercase
        term_lower = term.lower()

        # Remove specified phrases
        for phrase in self.phrases_to_remove:
            # Use word boundaries to avoid partial matches
            term_lower = re.sub(rf'\b{re.escape(phrase)}\b', '', term_lower, flags=re.IGNORECASE)

        # Remove punctuation
        term_cleaned = term_lower.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        term_cleaned = ' '.join(term_cleaned.split())

        return term_cleaned

    def clean_term_EM(self, term):
        """
        Specific cleaning for 'EM' annotation type, handling plural forms.

        Args:
            term (str): The term to clean.

        Returns:
            str: The cleaned term.
        """
        if term.endswith("es"):
            return term[:-2]
        elif term.endswith("s"):
            return term[:-1]
        return term

    def get_exact_match(self, term, term_dict):
        """
        Retrieve the exact match for a term from the term dictionary.

        Args:
            term (str): The term to match.
            term_dict (dict): Dictionary mapping terms to IDs.

        Returns:
            str or None: The matched term ID or None if no exact match is found.
        """
        return term_dict.get(term)

    def get_embedding_match(self, term, index, indexed_terms, term_dict, threshold=0.7):
        """
        Retrieve the best embedding match for a term using FAISS and cosine similarity.

        Args:
            term (str): The term to match.
            index (faiss.Index): FAISS index of term embeddings.
            indexed_terms (list): List of indexed terms corresponding to the FAISS index.
            term_dict (dict): Dictionary mapping terms to IDs.
            threshold (float): Minimum cosine similarity required to consider a match.

        Returns:
            str or None: The matched term ID or None if no suitable match is found.
        """
        # Obtain the embedding vector for the term
        term_doc = self.nlp(term)
        term_vector = term_doc.vector.reshape(1, -1).astype('float32')

        # Normalize the vector
        faiss.normalize_L2(term_vector)

        # Perform search on the FAISS index
        D, I = index.search(term_vector, 1)  # Retrieve the top 1 match

        if I[0][0] != -1:
            matched_term = indexed_terms[I[0][0]]
            matched_doc = self.nlp(matched_term)
            similarity = cosine_similarity(term_vector, matched_doc.vector.reshape(1, -1))[0][0]
            if similarity >= threshold:
                return term_dict.get(matched_term, None)
        return None

    def map_terms(self, entities, entity_type):
        """
        Map new entities using exact and embedding matches.

        Args:
            entities (list): List of entity strings to map.
            entity_type (str): The annotation type key (e.g., 'EM', 'Primer').

        Returns:
            dict: Dictionary mapping each entity to its matched term ID or "No Match".
        """
        if entity_type not in self.loaded_data:
            logging.error(f"Annotation type '{entity_type}' is not loaded.")
            return {entity: "No Match" for entity in entities}

        data = self.loaded_data[entity_type]
        term_dict = data["term_to_id"]
        indexed_terms = data["indexed_terms"]
        index = data["index"]

        mapped_entities = {}
        for entity in entities:
            cleaned_entity = self.clean_term(entity)
            match = self.get_exact_match(cleaned_entity, term_dict)
            if not match:
                if entity_type == 'DS':
                    cleaned = self.clean_term(entity)
                elif entity_type == 'EM':
                    cleaned = self.clean_term_EM(entity)
                else:
                    cleaned = cleaned_entity
                match = self.get_embedding_match(cleaned, index, indexed_terms, term_dict)
            mapped_entities[entity] = match if match else "No Match"
        return mapped_entities

    def map_terms_reverse(self, entities, entity_type):
        """
        Map entities using exact, similarity, and embedding matches,
        returning both code and term.

        Args:
            entities (list): List of entity strings to map.
            entity_type (str): The annotation type key.

        Returns:
            dict: Dictionary mapping each entity to a tuple (grounded_code, grounded_term).
        """
        if entity_type not in self.loaded_data:
            logging.error(f"Annotation type '{entity_type}' is not loaded.")
            return {entity: ("#", "#") for entity in entities}

        data = self.loaded_data[entity_type]
        term_dict = data["term_to_id"]
        id_to_term = data["id_to_term"]
        indexed_terms = data["indexed_terms"]
        index = data["index"]

        mapped_entities = {}
        for entity in entities:
            normalized_entity = entity.lower() if entity_type != 'GP' else entity
            match = self.get_exact_match(normalized_entity, term_dict)

            if not match:
                if entity_type == 'DS':
                    cleaned = self.clean_term(entity)
                elif entity_type == 'EM':
                    cleaned = self.clean_term_EM(entity)
                else:
                    cleaned = normalized_entity
                match = self.get_embedding_match(cleaned, index, indexed_terms, term_dict)

            if match:
                grounded_code = match
                grounded_term = id_to_term.get(match, "#")
            else:
                grounded_code, grounded_term = "#", "#"

            mapped_entities[entity] = (grounded_code, grounded_term)

        return mapped_entities

    def map_terms_batch(self, all_entities):
        """
        Map entities for each entity type using exact, fuzzy, and embedding matches.

        Args:
            all_entities (dict): Dictionary with keys as annotation types and values as lists of entities.

        Returns:
            tuple: (mapped_entities, unmatched_terms)
                - mapped_entities (dict): Mapped terms with annotation types.
                - unmatched_terms (list): List of entities that could not be matched.
        """
        mapped_entities = {}
        unmatched_terms = []

        for entity_type, entities in all_entities.items():
            if entity_type not in self.loaded_data:
                logging.warning(f"Annotation type '{entity_type}' is not loaded. Skipping.")
                continue

            data = self.loaded_data[entity_type]
            term_dict = data["term_to_id"]
            indexed_terms = data["indexed_terms"]
            index = data["index"]

            for entity in entities:
                cleaned_entity = self.clean_term(entity)
                match = self.get_exact_match(cleaned_entity, term_dict)
                if not match:
                    if entity_type == 'DS':
                        cleaned = self.clean_term(entity)
                    elif entity_type == 'EM':
                        cleaned = self.clean_term_EM(entity)
                    else:
                        cleaned = cleaned_entity
                    match = self.get_embedding_match(cleaned, index, indexed_terms, term_dict)

                if match:
                    mapped_entities[entity] = match
                else:
                    unmatched_terms.append({"term": entity})

        return mapped_entities, unmatched_terms

    def map_terms_reverse_batch(self, all_entities):
        """
        Map entities using exact, similarity, and embedding matches,
        returning both code and term.

        Args:
            all_entities (dict): Dictionary with keys as annotation types and values as lists of entities.

        Returns:
            tuple: (mapped_entities, unmatched_terms)
                - mapped_entities (dict): Mapped terms with annotation types.
                - unmatched_terms (list): List of entities that could not be matched.
        """
        mapped_entities = {}
        unmatched_terms = []

        for entity_type, entities in all_entities.items():
            if entity_type not in self.loaded_data:
                logging.warning(f"Annotation type '{entity_type}' is not loaded. Skipping.")
                continue

            data = self.loaded_data[entity_type]
            term_dict = data["term_to_id"]
            id_to_term = data["id_to_term"]
            indexed_terms = data["indexed_terms"]
            index = data["index"]

            for entity in entities:
                normalized_entity = entity.lower() if entity_type != 'GP' else entity
                match = self.get_exact_match(normalized_entity, term_dict)

                if not match:
                    if entity_type == 'DS':
                        cleaned = self.clean_term(entity)
                    elif entity_type == 'EM':
                        cleaned = self.clean_term_EM(entity)
                    else:
                        cleaned = normalized_entity
                    match = self.get_embedding_match(cleaned, index, indexed_terms, term_dict)

                if match:
                    grounded_code = match
                    grounded_term = id_to_term.get(match, "Unknown Term")
                    mapped_entities[entity] = (grounded_code, grounded_term)
                else:
                    unmatched_terms.append({"term": entity})

        return mapped_entities, unmatched_terms

    def map_to_url(self, entity_group, ent_id):
        """
        Map an entity ID to its corresponding URL based on the annotation type.

        Args:
            entity_group (str): The annotation type (e.g., 'EM', 'GO').
            ent_id (str): The entity ID to map.

        Returns:
            str: The corresponding URL or '#' if not found.
        """
        if not ent_id:
            return "#"
        if entity_group == 'EM':
            if ent_id.startswith("EFO"):
                return f"http://www.ebi.ac.uk/efo/{ent_id}"
            elif ent_id.startswith("MI") or ent_id.startswith("OBI"):
                return f"http://purl.obolibrary.org/obo/{ent_id}"
            else:
                return "#"
        elif entity_group == 'GO':
            ent_id = ent_id.replace('_', ':')
            return f"http://identifiers.org/{ent_id}"
        elif entity_group == 'primer':
            return f"http://probebase.csb.univie.ac.at/pb_report/probe/{ent_id}"

        # Handling for EFO (Experimental Factor Ontology)
        elif entity_group == 'EFO':
            # Special case: Entities starting with BFO or IAO are handled in the same way
            if ent_id.startswith(('EFO')):
                return f"http://www.ebi.ac.uk/efo/EFO_{ent_id}"
            else:
                return f"http://purl.obolibrary.org/obo/{ent_id}"

        # Default mappings
        url_map = {
            'GP': f"https://www.uniprot.org/uniprotkb/{ent_id}/entry",
            'DS': f"http://linkedlifedata.com/resource/umls-concept/{ent_id}",
            'OG': f"http://identifiers.org/taxonomy/{ent_id}",
            'CD': f"http://purl.obolibrary.org/obo/{ent_id}",
            'ENVO': f"http://purl.obolibrary.org/obo/{ent_id}"

        }
        return url_map.get(entity_group, "#")

