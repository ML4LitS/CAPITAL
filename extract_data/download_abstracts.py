import requests
import csv
from tqdm import tqdm


def get_unique_pmids(filename):
    pmids = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip the header
        for row in reader:
            pmids.append((row[0], row[2]))  # accession_type, id
    return pmids


def chunked(iterable, n):
    """Helper function to split the list into chunks"""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


url = "https://www.ebi.ac.uk/europepmc/webservices/rest/searchPOST"
pmids = get_unique_pmids("final_IDs_100000.csv")

def process_response(data, id_to_accession, csv_writer):
    """Function to process the API response and write to the CSV."""
    for article in data["resultList"]["result"]:
        abstract = article.get("abstractText", "").strip()  # Strip to remove leading/trailing whitespaces
        article_id = article["id"]
        corresponding_accession_type = id_to_accession.get(article_id)
        if corresponding_accession_type and abstract:  # Check if the abstract exists before writing
            csv_writer.writerow([article_id, corresponding_accession_type, abstract])



with open("final_abstracts_100000.csv", "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["ID", "accession_type", "abstract"])  # Header of CSV

    # Wrap the chunk iteration with tqdm for progress bar
    for chunk in tqdm(chunked(pmids, 500), desc="Processing PMCIDs"):
        # Separate the IDs based on type
        pmc_ids = [id_ for accession, id_ in chunk if id_.startswith('PMC')]
        med_ids = [id_ for accession, id_ in chunk if not id_.startswith('PMC')]

        # Create a dictionary for ID to accession_type mapping for the current chunk
        id_to_accession = {id_: accession for accession, id_ in chunk}

        # Query for PMC IDs
        if pmc_ids:
            query_pmc = " OR ".join([f"PMC:'{id_}'" for id_ in pmc_ids])
            query_params = {
                "query": query_pmc,
                "resultType": "core",
                "pageSize": len(pmc_ids),
                "format": "json"
            }
            response_pmc = requests.post(url, data=query_params)
            response_pmc.raise_for_status()  # Check for any errors in the response
            process_response(response_pmc.json(), id_to_accession, csv_writer)

        # Query for MED IDs
        if med_ids:
            query_med = " OR ".join([f"ext_id:{id_}" for id_ in med_ids])
            query_params = {
                "query": query_med,
                "resultType": "core",
                "pageSize": len(med_ids),
                "format": "json"
            }
            response_med = requests.post(url, data=query_params)
            response_med.raise_for_status()  # Check for any errors in the response
            process_response(response_med.json(), id_to_accession, csv_writer)

print("Data has been written to data_abstracts.csv")
