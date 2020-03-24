#
# minimap
#

import spacy
from spacy.tokens import Doc
from itertools import chain
import os
import robotreviewer
import pickle

# nlp = spacy.load("en")
nlp = spacy.load("en_core_web_sm")

# ignore list
with open(os.path.join(robotreviewer.DATA_ROOT, 'minimap', 'ignorelist.txt'), 'r') as f:
    ignores = set((l.strip() for l in f))


with open(os.path.join(robotreviewer.DATA_ROOT, 'minimap', 'str_to_cui.pck'), 'rb') as f:
    str_to_cui = pickle.load(f)


with open(os.path.join(robotreviewer.DATA_ROOT, 'minimap', 'cui_to_mh.pck'), 'rb') as f:
    cui_to_mh = pickle.load(f)


# add manual extras
with open(os.path.join(robotreviewer.DATA_ROOT, 'minimap', 'str_to_cui_supp.pck'), 'rb') as f:
    str_to_cui_supp = pickle.load(f)
str_to_cui.update(str_to_cui_supp)


with open(os.path.join(robotreviewer.DATA_ROOT, 'minimap', 'cui_to_mh_supp.pck'), 'rb') as f:
    cui_to_mh_supp = pickle.load(f)
cui_to_mh.update(cui_to_mh_supp)



# some extra filtering rules to improve precision

drop_terms = set()

for k, v in str_to_cui.items():
    # strings which are too ambiguous (too many CUIs... 15 from experimentation)
    if len(set(v))>15:
        drop_terms.add(k)


for k, v in str_to_cui.items():
    # strings which are too short to be informative (2 chars or less tends to generate nonsense CUIs)
    if len(k)<=2:
        drop_terms.add(k)

for t in drop_terms:
    str_to_cui.pop(t)


# regular expressions and text processing functions

import re

with open(os.path.join(robotreviewer.DATA_ROOT, 'minimap','prepositions_conjunctions.txt'), 'r') as f:
    prep_conj = [l.strip() for l in f]

prep_conj_re = re.compile(r'\b({})\b'.format('|'.join(prep_conj)))
nos_ignore = re.compile(r'\bNOS\b') # note do after lowercase
pos_ignore = re.compile(r"(?<=\w)(\'s?)\b")
left_paren = re.compile(r"^\[(X|V|D|M|EDTA|SO|Q)\]")
paren = re.compile(r"[\(\[]\w+[\)\]]")
strip_space = re.compile(r"\s+")

def remove_nos(text):
    return nos_ignore.sub(' ', text)

def remove_pos(text):
    return pos_ignore.sub('', text)

def syn_uninv(text):
    try:
        inversion_point = text.index(', ')
    except ValueError:
        # not found
        return text

    if inversion_point+2 == len(text):
        # i.e. if the ', ' is at the end of the string
        return text

    if prep_conj_re.search(text[inversion_point+2:]):
        return text
    else:
        return text[inversion_point+2:] + " " + text[:inversion_point]

def ne_parentheticals(text_str):
    text_str = left_paren.sub('', text_str)
    text_str = paren.sub('', text_str)
    return text_str

def get_lemma(t):
    if t.text in exceptions:
        return exceptions[t.text]
    else:
        return t.lemma_

# pipelines

def minimap(text_str, chunks=False, abbrevs=None):
    return matcher(pipeline(text_str, umls_mode=False, abbrevs=abbrevs), chunks=chunks)


def pipeline(text_str, umls_mode=True, abbrevs=None):

    # sub out abbreviations if abbreviation dict given
    if abbrevs:
        for abbrev, expansion in abbrevs.items():
            try:
                text_str = re.sub(r"\b" + re.escape(abbrev) + r"\b", expansion, text_str)

            except:
                print(f"Regex error caused for one abstract! (for text string '{text_str}')")
                print(f"and abbreviation dictionary '{abbrevs}'")
                # to avoid weird errors in abbreviations generating error causing regex strings (which are not causing a named exception)
                continue

    # 1. removal of parentheticals
    #     if umls_mode:
    text_str = ne_parentheticals(text_str)

    # hyphens to spaces
    text_str = text_str.replace('-', ' ')
    # 3. conversion to lowercase
    # text_str = text_str.lower()
    # 2. syntactic uninverstion
    if umls_mode:
        text_str = syn_uninv(text_str)
    # 4. stripping of possessives
    text_str = remove_pos(text_str)
    # strip NOS's
    if umls_mode:
        text_str = remove_nos(text_str)
    # last... remove any multiple spaces, or starting/ending with space
    text_str = strip_space.sub(' ', text_str)
    text_str = text_str.strip()
    return text_str



from itertools import chain


def matcher(text, chunks=False):
    doc = nlp(text.lower())

    if chunks:
        return list(chain.from_iterable(matcher(np.text, chunks=False) for np in doc.noun_chunks))
    tokens = [t.text.lower() for t in doc]
    lemmas = [t.lemma_ for t in doc if t.text.lower()]
    lemmas = [l for l in lemmas if l != '-PRON-']


    matches = []
    max_len = len(doc)
    window = max_len


    while window:

        for i in range(max_len - window + 1):
            window_text = ' '.join(tokens[i:i+window])
            window_lemma = ' '.join(lemmas[i:i+window])


            if window_lemma and window_lemma in str_to_cui and window_lemma not in ignores and window_text \
                not in nlp.Defaults.stop_words:


                for entry in str_to_cui[window_lemma]:
                    mh = cui_to_mh[entry].copy()
                    mh['start_idx'] = i
                    mh['end_idx'] = i+window
                    mh['source_text'] = doc[mh['start_idx']:mh['end_idx']].text
                    matches.append(mh)

        window -= 1



    matches.sort(key=lambda x: (x['start_idx'], -x['end_idx']))



    filtered_terms = []

    right_border = 0
    for match in matches:
        if match['start_idx'] >= right_border:
            filtered_terms.append(match)
            right_border = match['end_idx']

    return filtered_terms


def get_unique_terms(l, abbrevs=None):
    
    terms = [minimap(s, abbrevs=abbrevs) for s in l]
    flat_terms = [item for sublist in terms for item in sublist]
    encountered_terms = set()
    unique_terms = []
    for term in flat_terms:
        if term['cui'] not in encountered_terms:
            term.pop('start_idx')
            term.pop('end_idx')
            term.pop('source_text')
            unique_terms.append(term)
            encountered_terms.add(term['cui'])
    return unique_terms
    
