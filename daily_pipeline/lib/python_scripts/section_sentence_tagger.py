import os
import re
import gzip
import json

import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
from rapidfuzz import process, fuzz
import spacy
from fuzzywuzzy import fuzz, process


# Initialize spaCy model
nlp = spacy.load("en_core_sci_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
nlp.add_pipe("sentencizer")

# Precompile regex patterns for section tagging
titleMapsBody = {
    'INTRO': [
        'introduction', 'background', 'related literature', 'literature review', 'objective',
        'aim ', 'purpose of this study', 'study (purpose|aim|aims)', r'\d+\. (purpose|aims|aim)',
        '(aims|aim|purpose) of the study', '(the|drug|systematic|book) review', 'review of literature',
        'related work', 'recent advance', 'overview', 'i ntroduction', 'historical overview',
        'scope', 'context', 'rationale', 'hypothesis', 'motivation', 'i ntroduction', 'i ntro', 'i n t r o d u c t i o n'
    ],
    'METHODS': [
        'supplement', 'methods and materials', 'method', 'material', 'experimental procedure',
        'implementation', 'methodology', 'treatment', 'statistical analysis', "experimental",
        r'\d+\. experimental$', 'experimental (section|evaluation|design|approach|protocol|setting|set up|investigation|detail|part|perspective|tool)',
        "the study", r'\d+\. the study$', "protocol", "protocols", 'study protocol',
        'construction and content', r'experiment \d+', '^experiments$', 'analysis', 'utility',
        'design', r'\d+\. theory$', "theory", 'theory and ', 'theory of ',
        'data analysis', 'data collection', 'methodological approach', 'techniques', 'sample',
        'materials and methods', 'analytical methods', 'research methods', 'methodological framework',
        'm aterials and m ethods', 'm a t e r i a l s a n d m e t h o d s', 'm ethods'
    ],
    'RESULTS': [
        'result', 'finding', 'diagnosis', 'outcomes', 'findings', 'observations',
        'key results', 'main results', 'data', 'analysis results', 'primary results',
        'research findings', 'experimental results', 'empirical findings', 'report of results',
        'r esults', 'r e s u l t s'
    ],
    'DISCUSS': [
        'discussion', 'management of', r'\d+\. management', 'safety and tolerability',
        'limitations', 'perspective', 'commentary', r'\d+\. comment', 'interpretation',
        'interpretation of results', 'analysis of findings', 'discussion and implications',
        'contextualization', 'reflection', 'critical analysis', 'discussion and future work',
        'insights', 'consideration', 'comparison with previous studies', 'd iscussion', 'd i s c u s s i o n'
    ],
    'CONCL': [
        'conclusion', 'key message', 'future', 'summary', 'recommendation',
        'implications for clinical practice', 'concluding remark', 'closing remarks',
        'takeaway', 'final remarks', 'overall conclusion', 'summary and conclusion',
        'implications', 'closing statement', 'wrap-up', 'summary of findings',
        'future directions', 'outlook', 'next steps', 'c onclusion', 'c o n c l u s i o n'
    ],
    'CASE': [
        'case study report', 'case report', 'case presentation', 'case description',
        r'case \d+', r'\d+\. case', 'case summary', 'case history', 'case overview',
        'case study', 'case examination', 'case details', 'case documentation',
        'case example', 'case profile', 'c ase', 'c a s e'
    ],
    'ACK_FUND': [
        'funding', 'acknowledgement', 'acknowledgment', 'financial disclosure',
        'funding sources', 'funding support', 'financial support', 'grant support',
        'grant acknowledgement', 'acknowledgement of funding', 'funder', 'acknowledgements',
        'a c k n o w l e d g e m e n t', 'a c k f u n d'
    ],
    'AUTH_CONT': [
        "author contribution", "authors' contribution", "author's contribution",
        "contribution of authors", "authors' roles", "author responsibilities", "authorship contributions",
        'a u t h o r c o n t r i b u t i o n'
    ],
    'COMP_INT': [
        'competing interest', 'conflict of interest', 'conflicts of interest',
        'disclosure', 'declaration', 'competing interests', 'conflict statement',
        'financial conflicts', 'competing financial interests', 'c o m p i n t'
    ],
    'ABBR': [
        'abbreviation', 'abbreviations list', 'acronyms', 'nomenclature',
        'glossary', 'terms', 'terminology', 'abbreviation glossary', 'a b b r e v i a t i o n'
    ],
    'SUPPL': [
        'supplemental data', 'supplementary file', 'supplemental file', 'supplementary data',
        'supplementary figure', 'supplemental figure', 'supporting information',
        'supplemental file', 'supplemental material', 'supplementary material',
        'supplement material', 'additional data files', 'supplemental information',
        'supplementary information', 'supporting files', 'appendix', 'online appendix',
        'supporting documentation', 'extra data', 'additional material', 'annex',
        's u p p l e m e n t', 's u p p l e m e n t a r y'
    ]
}

titleExactMapsBody = {
    'INTRO': [
        "aim", "aims", "purpose", "purposes", "purpose/aim",
        "purpose of study", "review", "reviews", "minireview", "overview", "background",
        'i n t r o d u c t i o n', 'intro'
    ],
    'METHODS': [
        "experimental", "the study", "protocol", "protocols", "procedure", "methodology", "data analysis",
        'm e t h o d s', 'methods'
    ],
    'DISCUSS': [
        "management", "comment", "comments", "discussion", "limitations", "perspectives",
        'd i s c u s s', 'discussion'
    ],
    'CASE': [
        "case", "cases", "case study", "case report", "case overview", 'case'
    ]
}

titleMapsBack = {
    'REF': [
        'reference', 'literature cited', 'references', 'bibliography', 'source list', 'citations',
        'works cited', 'cited literature', 'bibliographical references', 'citations list',
        'r e f e r e n c e s'
    ],
    'ACK_FUND': [
        'funding', 'acknowledgement', 'acknowledgment', 'acknowlegement',
        'acknowlegement', 'open access', 'financial support', 'grant',
        'author note', 'financial disclosure', 'support statement', 'funding acknowledgment',
        'a c k n o w l e d g e'
    ],
    'ABBR': [
        'abbreviation', 'glossary', 'abbreviations list', 'acronyms', 'terminology', 'abbreviation glossary',
        'a b b r e v i a t i o n'
    ],
    'COMP_INT': [
        'competing interest', 'conflict of interest', 'conflicts of interest',
        'disclosure', 'declaration', 'conflicts', 'interest', 'financial conflicts',
        'c o m p i n t'
    ],
    'CASE': [
        'case study report', 'case report', 'case presentation', 'case description',
        r'case \d+', r'\d+\. case', 'case summary', 'case history', 'case overview',
        'case study', 'case examination', 'case details', 'case documentation',
        'case example', 'case profile', 'c ase', 'c a s e'
    ],
    'ACK_FUND': [
        'funding', 'acknowledgement', 'acknowledgment', 'financial disclosure',
        'funding sources', 'funding support', 'financial support', 'grant support',
        'grant acknowledgement', 'acknowledgement of funding', 'funder', 'acknowledgements',
        'a c k n o w l e d g e m e n t', 'a c k f u n d'
    ],
    'AUTH_CONT': [
        "author contribution", "authors' contribution", "author's contribution",
        "contribution of authors", "authors' roles", "author responsibilities", "authorship contributions",
        'a u t h o r c o n t r i b u t i o n'
    ],
    'COMP_INT': [
        'competing interest', 'conflict of interest', 'conflicts of interest',
        'disclosure', 'declaration', 'competing interests', 'conflict statement',
        'financial conflicts', 'competing financial interests', 'c o m p i n t'
    ],
    'ABBR': [
        'abbreviation', 'abbreviations list', 'acronyms', 'nomenclature',
        'glossary', 'terms', 'terminology', 'abbreviation glossary', 'a b b r e v i a t i o n'
    ],
    'SUPPL': [
        'supplemental data', 'supplementary file', 'supplemental file', 'supplementary data',
        'supplementary figure', 'supplemental figure', 'supporting information',
        'supplemental file', 'supplemental material', 'supplementary material',
        'supplement material', 'additional data files', 'supplemental information',
        'supplementary information', 'supporting files', 'appendix', 'online appendix',
        'supporting documentation', 'extra data', 'additional material', 'annex',
        's u p p l e m e n t', 's u p p l e m e n t a r y'
    ]
}

ordered_labels = ['TITLE', 'ABSTRACT', 'INTRO', 'METHODS', 'RESULTS', 'DISCUSS', 'CONCL', 'CASE',
                  'ACK_FUND', 'AUTH_CONT', 'COMP_INT', 'ABBR', 'SUPPL', 'REF', 'ACK_FUND', 'ABBR',
                  'COMP_INT', 'SUPPL', 'APPENDIX', 'AUTH_CONT']


# Precompile regex patterns
compiled_titleMapsBody = {
    key: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for key, patterns in titleMapsBody.items()
}

compiled_titleExactMapsBody = {
    key: [pattern.lower() for pattern in patterns]
    for key, patterns in titleExactMapsBody.items()
}

compiled_titleMapsBack = {
    key: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for key, patterns in titleMapsBack.items()
}


def createSecTag(soup, secType):
    secTag = soup.new_tag('SecTag')
    secTag['type'] = secType
    return secTag


# Function to read XML or GZ files and split into individual articles
def getfileblocks(file_path, document_flag):
    sub_file_blocks = []
    if file_path.endswith('.gz'):
        open_func = lambda x: gzip.open(x, 'rt', encoding='utf8')
    else:
        open_func = lambda x: open(x, 'r', encoding='utf8')

    try:
        with open_func(file_path) as fh:
            content = fh.read()
            if document_flag in ['f', 'a']:
                # Split content by <!DOCTYPE article ...> or <article ...> tags
                articles = re.split(r'(?=<!DOCTYPE article|<article(?![\w-]))', content)
                sub_file_blocks = [article.strip() for article in articles if
                                   article.strip() and '<!DOCTYPE' not in article]
            else:
                print('ERROR: unknown document type :' + document_flag)
    except Exception as e:
        print('Error processing file: ' + str(file_path))
        print(e)

    return sub_file_blocks


# Function to split text into sentences using spaCy
def sentence_split(text):
    sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        sentences.append(sent.text.strip())
    return sentences


# Function to process nested tags and collect sentences
def call_sentence_tags(ch):
    sentences = []
    for gch in ch.children:
        if isinstance(gch, str):
            continue  # Skip strings directly under ch
        if gch.name in ['article-title', 'title', 'subtitle', 'trans-title', 'trans-subtitle', 'alt-title', 'label',
                        'td', 'th']:
            if gch.find('p', recursive=False):
                sub_sentences = call_sentence_tags(gch)
                sentences.extend(sub_sentences)
            else:
                text = gch.get_text(separator=' ', strip=True)
                if text:
                    sents = sentence_split(text)
                    sentences.extend(sents)
        elif gch.name in ["sec", "fig", "statement", "div", "boxed-text", "list", "list-item", "disp-quote", "speech",
                          "fn-group", "fn", "def-list", "def-item", "def", "ack", "array", "table-wrap", "table",
                          "tbody", "thead", "tr", "caption", "answer", "sec-meta", "glossary", "question",
                          "question-wrap"]:
            sub_sentences = call_sentence_tags(gch)
            sentences.extend(sub_sentences)
        elif gch.name == 'p':
            sub_sentences = process_p_tag(gch)
            sentences.extend(sub_sentences)
        else:
            text = gch.get_text(separator=' ', strip=True)
            if text:
                sents = sentence_split(text)
                sentences.extend(sents)
    return sentences


# Function to process paragraph tags
def process_p_tag(gch):
    sentences = []
    if not (len(gch.contents) == 1 and (not gch.contents[0].string) and (
            gch.contents[0].name in ["ext-link", "e-mail", "uri", "inline-supplementary-material", "related-article",
                                     "related-object", "address", "alternatives", "array", "funding-source",
                                     "inline-graphic"])):
        text = gch.get_text(separator=' ', strip=True)
        if text:
            sents = sentence_split(text)
            sentences.extend(sents)
    return sentences


# Function to process the front section
def process_front(front):
    sections = {}
    keywords = []

    if front.find('article-meta'):
        art_meta = front.find('article-meta')

        for ch in art_meta.find_all(recursive=False):
            if ch.name in ['title-group', 'supplement', 'supplementary-material', 'abstract', 'trans-abstract',
                           'kwd-group', 'funding-group']:
                section_title = ch.name.upper()

                if section_title == 'KWD-GROUP':
                    # Extract keywords as a list from kwd-group
                    keywords = [kwd.text.strip() for kwd in ch.find_all('kwd')]
                else:
                    sentences = call_sentence_tags(ch)
                    if sentences:
                        sections.setdefault(section_title, []).extend(sentences)
            else:
                pass  # Ignore other tags

    return sections, keywords


# Function to process the back section
def process_back(back):
    sections = {}
    for ch in back.find_all(recursive=False):
        if ch.name in ['sec', 'p', 'ack', 'alternatives', 'array', 'preformat', 'fig', 'fig-group', 'question-wrap',
                       'question-wrap-group', 'list', 'table-wrap-group', 'table-wrap', 'display-formula',
                       'display-formula-group', 'def-list', 'list', 'supplementary-material', 'kwd-group',
                       'funding-group', 'statement', 'ref-list', 'glossary']:
            # Sections with titles
            if ch.name == 'ref-list':
                sentences = reference_sents(ch)
                if sentences:
                    sections.setdefault('REF', []).extend(sentences)
            else:
                title = ch.find('title')
                if title:
                    section_title = title.get_text(separator=' ', strip=True).strip().upper()
                else:
                    section_title = ch.name.upper()
                sentences = call_sentence_tags(ch)
                if sentences:
                    sections.setdefault(section_title, []).extend(sentences)
        else:
            pass  # Ignore other tags
    return sections


# Function to process reference sentences
def reference_sents(ref_list):
    sentences = []
    for ch in ref_list.children:
        if isinstance(ch, str):
            continue  # Skip strings directly under ref_list
        if ch.name == 'ref':
            sub_text = ''
            for gch in ch.children:
                if isinstance(gch, str):
                    continue
                sub_text += " " + " ".join([d.string for d in gch.descendants if d.string])
            if sub_text:
                sents = sentence_split(sub_text)
                sentences.extend(sents)
        elif ch.name in ["sec", "fig", "statement", "div", "boxed-text", "list", "list-item", "disp-quote", "speech",
                         "fn-group", "fn", "def-list", "def-item", "def", "ack", "array", "table-wrap", "table",
                         "tbody", "caption", "answer", "sec-meta", "glossary", "question", "question-wrap"]:
            sub_sentences = call_sentence_tags(ch)
            sentences.extend(sub_sentences)
        else:
            pass  # Ignore other tags
    return sentences


# Function to match section titles to predefined section types
def titleMatch(title, secFlag):
    matchKeys = []
    # Check if the flag is 'body' or 'back' and apply the respective mappings
    if secFlag == 'body':
        titleMaps = compiled_titleMapsBody
        exactMaps = compiled_titleExactMapsBody
    else:
        titleMaps = compiled_titleMapsBack
        exactMaps = {}

    title_lower = title.lower().strip()
    # Check exact matches first
    for key, patterns in exactMaps.items():
        if title_lower in patterns:
            matchKeys.append(key)
            break  # If exact match found, no need to check further

    # If no exact match, check regex patterns
    if not matchKeys:
        for key, patterns in titleMaps.items():
            if any(pattern.search(title_lower) for pattern in patterns):
                matchKeys.append(key)

    return ','.join(matchKeys) if matchKeys else None


# Function to apply section tagging to the soup object
def section_tag(soup):
    # Add Figure sections
    for fig in soup.find_all('fig', recursive=True):
        if not fig.find_all('fig', recursive=True):
            fig_tag = createSecTag(soup, 'FIG')
            fig.wrap(fig_tag)

    # Add Table sections
    for table in soup.find_all('table-wrap', recursive=True):
        if not table.find_all('table-wrap', recursive=True):
            table_tag = createSecTag(soup, 'TABLE')
            table.wrap(table_tag)

    # Process front section
    if soup.front:
        if soup.front.abstract:
            secAbs = createSecTag(soup, 'ABSTRACT')
            soup.front.abstract.wrap(secAbs)
        if soup.front.find('kwd-group'):
            secKwd = createSecTag(soup, 'KEYWORD')
            soup.front.find('kwd-group').wrap(secKwd)

    # Process body section
    if soup.body:
        for sec in soup.body.find_all('sec', recursive=False):
            title = sec.find('title')
            if title:
                title_text = title.get_text(separator=' ', strip=True)
                mappedTitle = titleMatch(title_text, 'body')
                if mappedTitle:
                    secBody = createSecTag(soup, mappedTitle)
                    sec.wrap(secBody)
    # Process back sections
    if soup.back:
        for sec in soup.back.find_all(['sec', 'ref-list', 'app-group', 'ack', 'glossary', 'notes', 'fn-group'],
                                      recursive=False):
            if sec.name == 'ref-list':
                secRef = createSecTag(soup, 'REF')
                sec.wrap(secRef)
            else:
                title = sec.find('title')
                if title:
                    title_text = title.get_text(separator=' ', strip=True)
                    mappedTitle = titleMatch(title_text, 'back')
                    if mappedTitle:
                        secBack = createSecTag(soup, mappedTitle)
                        sec.wrap(secBack)


# Function to process the body section
def process_body(body):
    sections = {}
    for ch in body.find_all(recursive=False):
        if ch.name == 'p':
            sentences = process_p_tag(ch)
            sections.setdefault('BODY', []).extend(sentences)
        elif ch.name in ['sec', 'ack', 'alternatives', 'array', 'preformat', 'fig', 'fig-group', 'question-wrap',
                         'list', 'table-wrap-group', 'table-wrap', 'display-formula', 'display-formula-group',
                         'def-list', 'list', 'supplementary-material', 'kwd-group', 'funding-group', 'statement']:
            title = ch.find('title')
            if title:
                section_title = title.get_text(separator=' ', strip=True).strip().upper()
            else:
                section_title = ch.name.upper()
            sentences = call_sentence_tags(ch)
            if sentences:
                sections.setdefault(section_title, []).extend(sentences)
    return sections


# Main function to process each article and collect data
def process_full_text(each_file):
    # Replace body tag with orig_body to prevent BeautifulSoup from removing it
    each_file = re.sub(r'<body(\s[^>]*)?>', '<orig_body\\1>', each_file)
    each_file = each_file.replace('</body>', '</orig_body>')
    try:
        xml_soup = BeautifulSoup(each_file, 'lxml')
        # Remove extra html and body tags added by BeautifulSoup
        if xml_soup.html:
            xml_soup.html.unwrap()
        if xml_soup.body:
            xml_soup.body.unwrap()
        if xml_soup.find('orig_body'):
            xml_soup.find('orig_body').name = 'body'

        # Extract attributes from the <article> tag
        article_tag = xml_soup.find('article')
        if article_tag:
            open_status = article_tag.get('open-status', '')
            article_type = article_tag.get('article-type', '')
        else:
            open_status = ''
            article_type = ''

        # Extract article IDs
        article_ids = {}
        for id_tag in xml_soup.find_all('article-id'):
            id_type = id_tag.get('pub-id-type', 'unknown')
            article_ids[id_type] = id_tag.text.strip()
        if not article_ids:
            print('No article IDs found')
            return None

        # # print(article_ids)
        # if 'pmcid' not in article_ids or article_ids.get('pmcid') != '11376101':
        #     return None
        # Apply section tagging
        section_tag(xml_soup)
        # print(xml_soup)
        sections = {}
        keywords = []

        # Process sections under SecTag
        for sec_tag in xml_soup.find_all('SecTag'):
            sec_type = sec_tag.get('type', 'unknown').strip().upper()
            if sec_type == 'KEYWORD':
                # Extract keywords
                keywords = [kwd.text.strip() for kwd in sec_tag.find_all('kwd')]
                continue  # Skip further processing of keywords here
            if sec_type not in sections:
                sections[sec_type] = []
            # Exclude nested 'SecTag's to avoid duplicate text
            for nested_sec in sec_tag.find_all('SecTag', recursive=True):
                nested_sec.extract()
            sentences = call_sentence_tags(sec_tag)
            sections[sec_type].extend(sentences)

        # Process front section if not already processed
        if xml_soup.article.find('front'):
            front_sections, front_keywords = process_front(xml_soup.article.find('front'))
            for k, v in front_sections.items():
                sections.setdefault(k, []).extend(v)
            if front_keywords:
                keywords.extend(front_keywords)

        # Process body section if not already processed
        if xml_soup.article.find('body'):
            body_sections = process_body(xml_soup.article.find('body'))
            for k, v in body_sections.items():
                sections.setdefault(k, []).extend(v)

        # Process back section if not already processed
        if xml_soup.article.find('back'):
            back_sections = process_back(xml_soup.article.find('back'))
            for k, v in back_sections.items():
                sections.setdefault(k, []).extend(v)

        # Remove empty sections
        sections = {k: v for k, v in sections.items() if v}

        return {
            'article_ids': article_ids,
            'open_status': open_status,
            'article_type': article_type,
            'keywords': keywords,
            'sections': sections
        }

    except Exception as e:
        print(f"Error processing article: {e}")
        return None


# from fuzzywuzzy import fuzz, process

def process_json(data, ordered_labels):
    if not data or 'sections' not in data:
        print("Invalid data or missing 'sections'")
        return {}

    if not ordered_labels:
        print("No ordered labels provided for matching.")
        return {}

    # Step 1: Initialize sections and directly map TITLE-GROUP to TITLE if present
    sections = data['sections']
    if "TITLE-GROUP" in sections:
        sections["TITLE"] = sections.pop("TITLE-GROUP")

    # Step 2: Identify keys in sections not present in ordered_labels
    section_keys = set(sections.keys())
    ordered_labels_set = set(ordered_labels)
    unfound_keys = section_keys - ordered_labels_set  # Keys in sections not in ordered_labels

    # Step 3: Normalize only the unfound section keys (remove spaces, uppercase)
    normalized_unfound_keys = {key.replace(" ", "").upper(): key for key in unfound_keys}

    # Step 4: Map unfound normalized keys to ordered labels using fuzzy matching (threshold 80%)
    mapped_labels = {}
    for normalized_key, original_key in normalized_unfound_keys.items():
        if not normalized_key:
            mapped_labels[original_key] = "OTHER"
            continue  # Skip fuzzy matching for empty keys

        # Perform fuzzy matching
        match, score = process.extractOne(normalized_key, ordered_labels, scorer=fuzz.partial_ratio)
        if not match or score < 80:
            mapped_labels[original_key] = "OTHER"  # Use "OTHER" for no close match
        else:
            mapped_labels[original_key] = match

    # # Step 6: Structure JSON without ordering or sent_id for now
    result_json = {}
    for section_key in sections:
        label = mapped_labels.get(section_key, section_key)  # Use mapped label if exists, else original
        texts = [{"text": text} for text in sections[section_key]]
        if label in result_json:
            result_json[label].extend(texts)  # Append to existing list
        else:
            result_json[label] = texts  # Create new list

    # Step 6: Reorder JSON according to ordered_labels and add any unmapped sections at the end
    ordered_json = {}
    for label in ordered_labels:
        if label in result_json:
            ordered_json[label] = result_json.pop(label)
    ordered_json.update(result_json)  # Add remaining sections in their original order

    # Step 7: Assign unique incremental sent_id starting from 1
    sent_id = 1
    for section in ordered_json.values():
        for entry in section:
            entry["sent_id"] = sent_id
            sent_id += 1  # Increment sent_id for each entry uniquely

    # Step 7: Combine reordered sections with metadata
    combined_data = {
        'article_ids': data['article_ids'],
        'open_status': data['open_status'],
        'article_type': data['article_type'],
        'keywords': data['keywords'],
        'sections': ordered_json
    }

    return combined_data


# Function to process each article and write to a compressed output file
def process_each_article(each_file_path, out_file, document_flag):
    files_list = getfileblocks(each_file_path, document_flag)

    # Open the output file in gzip write mode
    with gzip.open(out_file, 'wt', encoding='utf-8') as out:
        for each_file in tqdm(files_list, desc="Processing Articles", disable=False):
            if document_flag == 'f':
                data_temp = process_full_text(each_file)
                if data_temp:
                    # print(data_temp)
                    data = process_json(data_temp, ordered_labels)
                    # print(data)
                else:
                    print("Skipping file: data_temp is None")
                    continue
            else:
                print('Document type not supported.')
                continue
            if data:
                out.write(json.dumps(data) + '\n')


# Entry point
if __name__ == '__main__':
    process_each_article("/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch-28-10-2024-0.xml.gz","/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch-28-10-2024-0.jsonl.gz", "f")
    # parser = argparse.ArgumentParser(description='Process XML files and output sentences and sections.')
    # parser.add_argument('--input', help='Input XML or GZ file path', required=True)
    # parser.add_argument('--output', help='Output JSONL file path', required=True)
    # parser.add_argument('--type', help='Document type: f for full text, a for abstract', choices=['f', 'a'], required=True)
    # args = parser.parse_args()
    #
    # # Call process_each_article with the full output file path
    # process_each_article(args.input, args.output, args.type)

