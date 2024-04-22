import os
import requests
import csv
from tqdm import tqdm

# Constants
API_ENDPOINT = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsBySectionAndOrType"
PAGE_SIZE = 8

# Open types.txt and read the types
with open('types.txt', 'r') as f:
    types = f.read().splitlines()

# Find the index of 'ensembl' and slice the list to get only types after it
# try:
#     ensembl_index = types.index('ensembl')
#     types = types[ensembl_index + 1:]
# except ValueError:  # if 'ensembl' is not in the list, process all types
#     pass

# Determine if we need to write headers to the CSV
write_headers = not os.path.exists('output.csv') or os.path.getsize('output.csv') == 0

# Open the CSV file to write data
with open('output.csv', 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    if write_headers:
        csv_writer.writerow(['accession_type', 'source', 'extid', 'pmcid'])  # Write CSV headers only if required

    for type_ in tqdm(types, desc="Processing", unit="type"):
        cursor_mark = 0

        while True:
            params = {
                "type": "Accession Numbers",
                "subType": type_,
                "filter": 1,
                "format": "ID_LIST",
                "cursorMark": cursor_mark,
                "pageSize": PAGE_SIZE
            }

            response = requests.get(API_ENDPOINT, params=params)
            data = response.json()

            # Ensure that 'articles' key exists in the data
            if 'articles' not in data:
                print(f"Warning: 'articles' key not found for type: {type_}. Skipping...")
                break

            for article in data['articles']:
                source = article.get('source', '')
                extId = article.get('extId', '')
                pmcid = article.get('pmcid') or (
                    article.get('fullTextIdList')[0] if 'fullTextIdList' in article and article[
                        'fullTextIdList'] else '')
                csv_writer.writerow([type_, source, extId, pmcid])

            # Check if we have reached the end of the data
            if cursor_mark == data.get('nextCursorMark', None):
                break

            cursor_mark = data.get('nextCursorMark', 0)

print("Data has been appended to output.csv")


