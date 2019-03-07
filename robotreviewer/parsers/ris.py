"""
Simple RIS file parser

Tested and works with the dialects used by PubMed (MEDLINE export format) and Ovid (EndNote export format)

Exercise caution with other RIS formats or it might go wrong :)

"""
import codecs
from collections import OrderedDict
import re

def iter_load_ris(iterable):
    """
    takes either a file or list
    line-by-line
    """

    ris_re = re.compile('^[A-Z0-9]{1,4}\s*\-\s')

    delim_ovid_en = re.compile('^\<[1-9][0-9]*\. \>') # endnote export
    delim_ovid_ris = re.compile('^[1-9][0-9]*\.') # ris export
    delim_pubmed = re.compile('^\s*$')

    # additional useless stuff in Wiley RIS format
    wiley_ignores = [re.compile('^Record \#[1-9]+[0-9]* of [1-9]+[0-9]*$'),
                     re.compile('^Provider: John Wiley & Sons, Ltd.$'),
                     re.compile('^Content: text\/plain\; charset\=\"UTF\-8\"$')]




    needle_down = False # record player metaphor...
    entry_builder = OrderedDict()

    first_line = True

    for line in iterable:

        # first check for BOM from different file formats and remove if needed
        if line.startswith(codecs.BOM_UTF8.decode()):
            print("defused a BOM")
            line = line[1:]

        # skip any Wiley extra stuff
        if any((wi.match(line) for wi in wiley_ignores)):
            continue

        if first_line:
            first_line = False
            # infer a format from the first line
            if delim_ovid_en.match(line):
                delim = delim_ovid_en
                print("ovid endnote format")
            elif delim_ovid_ris.match(line):
                print("ovid ris")
                delim = delim_ovid_ris
            elif delim_pubmed.match(line):
                # i.e. if it starts with a blank line, or straight into the format
                print("pubmed")
                delim = delim_pubmed
            elif ris_re.match(line):
                print("other non numbered")
                delim = delim_pubmed


        if needle_down == False and ris_re.match(line):
            needle_down = True

        elif needle_down == True and delim.match(line):
            new_entry = OrderedDict()
            for k, v in entry_builder.items():
                new_entry[k] = v
            yield(new_entry)
            entry_builder = OrderedDict()
            needle_down = False

        if needle_down:
            if ris_re.match(line):
                key = line[:4].rstrip()
                value = line[6:].rstrip()
            else:
                key = last_key
                value = line.strip()

            if key not in entry_builder:
                # since using an ordered dict can't do quicker
                entry_builder[key] = []

            entry_builder[key].append(value)
            last_key = key

    if entry_builder:
        new_entry = OrderedDict()
        for k, v in entry_builder.items():
            new_entry[k] = v
        yield(new_entry)






def load(ris_file_obj):
    return [i for i in iter_load_ris(ris_file_obj)]

def loads(ris_string):
    return [i for i in iter_load_ris(ris_string.splitlines())]

def loadf(ris_filename):
    with open(ris_filename, 'r') as f:
        out = load(f)
    return out

def dumps(ris_list):

    out = []

    for article in ris_list:
        for k, v_list in article.items():
            if isinstance(v_list, list):
                for v in v_list:
                    out.append('{}  - {}'.format(k, v))
            elif any((isinstance(v_list, typ) for typ in [str, int, bool, float])):
                out.append('{}  - {}'.format(k, v_list))
        out.append('\n\n\n')

    return '\n'.join(out)

def dump(ris_list, file_obj):
    file_obj.write(dumps(ris_list))

def simplify(article):

    # Note currently the article *must* have at least a title
    # otherwise an exception happens

    # When strict=True, exceptions are raised
    # else empty data returned with a status message field

    try:
        # this should work for both PubMed and the Ovid/Endnote format
        out = {"title": ' '.join(article.get('TI', article.get('T1', []))),
               "abstract": ' '.join(article.get('AB', []))}
        if 'PT' in article:
            out['ptyp'] = article['PT']
        # have an explicit use_ptyp variable, which is automatically 
        # detected
        # for PubMed, this will autoset to True for where
        # status = "MEDLINE". (and hence MeSH tagging is complete)
        # for Ovid, this will just check whether the MEDLINE database
        # was used (and if so, we have MeSH for all)
        # for all other options, we will only use title/abstract

        
        if "MEDLINE" in article.get('STAT', []):
            # PubMed + MEDLINE article
            out['use_ptyp'] = True
        elif "Ovid MEDLINE(R)" in article.get('DB', []):
            # Ovid + MEDLINE article
            out['use_ptyp'] = True
        else:
            out['use_ptyp'] = False
    except:
        raise Exception('Data was not recognised as Ovid or PubMed format')
    return out



