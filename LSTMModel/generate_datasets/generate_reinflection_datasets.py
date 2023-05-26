import os
from os.path import join
from argparse import ArgumentParser
import codecs
from tqdm import tqdm
import re
import random
import numpy as np
random.seed(42)
np.random.seed(42)

def filter_dict(keys, d:dict):
    return { k: d[k] for k in keys }

def handle_georgian(data):
    """
    A specific function for Georgian: ignore the verbs of the old file ("kat.txt") and take verbs only from the new file ("katVerbsNew.txt").
    *The new file contains only verbs
    """
    data = list(filter(lambda x: not x[2].startswith('V;'), data)) # filter old verbs
    new_verbs = dict2lists(read(join("../.data", "RawData", "katVerbsNew.txt"), 'kat'))
    new_data = data+new_verbs
    return new_data


def read(fname, lang:str):
    """ read file name """
    D = {}
    with codecs.open(fname, 'rb', encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            line = line.strip()
            if line in {'', ' '}: continue
            lemma, word, tag = line.split("\t")
            if lang=='hun' and word=="intransitive verb, definite forms are not used": continue # filters 712 bad forms
            if bool(re.search(r'\d', lemma+word)):  # filter all the words that contain numbers
                print(f"Contains digits: {fname[:3]}, line {i}: {lemma, word, tag}")
                continue
            if lemma not in D:
                D[lemma] = {}
            D[lemma][tag] = word
    return D

def dict2lists(d):
    """
    Convert the given defaultdict object to a list of (lemma,form,feat) tuples.
    d is a list of tuples, where wach tuple has the form (lemma, list) where list has (at most?) 3 dicts with items of the form (feat,form).
    """
    samples_list = []
    for lemma, pairs in d.items(): # t is a tuple
        for k,v in pairs.items():
            samples_list.append((lemma, v, k))
    return samples_list

def lists2dict(lists):
    # Takes a list of tuples and stores the tables as dictionary of dictionaries
    D = {}
    for l in lists:
        lemma, form, feat = l
        if lemma not in D:
            D[lemma] = {}
        D[lemma][feat] = form
    return D


def write_dataset(p, dataset, message=''):
    """
    Takes the new dataset, and writes it to a given new path.
    """
    if message!='': print(message)
    open(p, mode='w', encoding='utf8').write('\n'.join(['\t'.join(item) for item in dataset]))


def generate_reinflection(data:dict, num_samples:int, message=''):
    # Generate distinct reinflection samples from the dict
    if message!='': print(message)
    lemmas = list(data.keys())
    lemmas_indices = np.random.choice(range(len(lemmas)), num_samples)
    samples = [(-1, -1)]
    for i,lemma_idx in tqdm(enumerate(lemmas_indices)):
        lemma = lemmas[lemma_idx]
        form_idx1, form_idx2 = -1, -1
        while (form_idx1, form_idx2) in samples:
            form_idx1, form_idx2 = np.random.choice(range(len(data[lemma])), 2, replace=False)
        lemma_dict_items = dict(enumerate(data[lemma].items()))
        src_tuple, trg_tuple = lemma_dict_items[form_idx1], lemma_dict_items[form_idx2]
        samples.append(src_tuple + trg_tuple)
    del samples[0] # remove the (-1, -1)
    return samples


def split_lines_by_POS(data):
    # Assuming the possible POSs are 'V', 'N' and 'ADJ'.
    pos_data = {'V':[], 'N':[], 'ADJ':[]}
    for i,line in enumerate(data,1):
        lemma, form, feat = line
        features = feat.split(';')
        if features[0] not in {'V', 'N', 'ADJ'}:
            if features[0].startswith("V."): # it's a badly-annotated verb
                features.insert(0, 'V')
                line = (lemma, form, feat.replace('V.',"V;V."))
            else:
                raise Exception(f"Unidentified badly-annotated line (#{i}): {lemma, form, feat}")
        pos_data[features[0]].append(line)
    actual_POS = dict(filter(lambda x: x[1]!=[], pos_data.items())) # filter empty lists
    return actual_POS


def main():
    parser = ArgumentParser(description="Generate Train-Dev-Test Reinflection sets for a given language.")
    parser.add_argument('lang', type=str, choices=['kat', 'tur', 'fin', 'bul', 'hun', 'lav', 'swc', 'sqi'], help="Language to be processed")
    parser.add_argument('dataset_size', type=int, default=10000, help="The total number of samples, which will be split to train-dev-test in 0.8-0.1-0.1 ratios")
    args = parser.parse_args()
    lang = args.lang
    dataset_size = args.dataset_size
    train_index, dev_index, dev_size = int(0.8*dataset_size), int(0.9*dataset_size), int(0.1*dataset_size)
    # The file of the inflection tables, possibly with several POS.
    file_path = join("../.data", "RawData", f"{lang}.txt")
    # 1. Separate the data to the POSs.
    data_dict = read(file_path, lang)
    data_lists = dict2lists(data_dict)
    if lang=='kat': data_lists = handle_georgian(data_lists)
    data_POS = split_lines_by_POS(data_lists)
    # 2. Write the split data to inflection tables
    if not os.path.isdir(join("../.data", "InflectionTables")): os.mkdir(join("../.data", "InflectionTables"))
    if not os.path.isdir(join("../.data", "Reinflection")): os.mkdir(join("../.data", "Reinflection"))
    for pos,pos_list in data_POS.items():
        new_path = join("../.data", "InflectionTables", f"{lang}.{pos}.txt")
        write_dataset(new_path, pos_list, message=f"Writing the file {new_path}")

    # 3. For each pos, store all the tables in a table
    data_POS_dict = {k:lists2dict(l) for k,l in data_POS.items()}
    for pos, pos_dictionary in data_POS_dict.items():
        # 4. Generate form-split samples
        reinflection_dir = join("../.data", "Reinflection", f"{lang}.{pos}")
        if not os.path.isdir(reinflection_dir): os.mkdir(reinflection_dir)

        form_split = generate_reinflection(pos_dictionary, num_samples=dataset_size, message=f"Generating {lang}.{pos} form-split data:")
        form_split_train_dev_test = [form_split[:train_index], form_split[train_index:dev_index], form_split[dev_index:]]
        print("Writing form-split reinflection samples")
        for name,subset in zip(['train','dev','test'], form_split_train_dev_test):
            fname = join(reinflection_dir, f"{lang}.{pos}.form.{name}.txt")
            write_dataset(fname, subset, message=f"Writing the {name} file {fname}")

        # 5. generate lemma-split samples
        if len(pos_dictionary)<10:
            print(f"Found for {lang}.{pos} only {len(pos_dictionary)} tables, therefore lemma-split datasets are impossible to generate! Must supply at least 10 tables!")
            continue
        lemmas = list(pos_dictionary.keys())
        np.random.shuffle(lemmas)

        train_lemmas_index, dev_lemmas_index = int(0.8*len(lemmas)), int(0.9*len(lemmas))
        lemmas_train_dev_test = [lemmas[:train_lemmas_index], lemmas[train_lemmas_index:dev_lemmas_index], lemmas[dev_lemmas_index:]]
        train_dict, dev_dict, test_dict = [filter_dict(lemmas_subset, pos_dictionary) for lemmas_subset in lemmas_train_dev_test] # take the chosen lemmas of each set

        lemma_train = generate_reinflection(train_dict, num_samples=train_index, message=f"Generating {lang}.{pos} lemma-split train data:")
        lemma_dev = generate_reinflection(dev_dict, num_samples=dev_size, message=f"Generating {lang}.{pos} lemma-split dev data:")
        lemma_test = generate_reinflection(test_dict, num_samples=dev_size, message=f"Generating {lang}.{pos} lemma-split test data:") # same size as dev size

        print("Writing form-split reinflection samples")
        for name,subset in zip(['train','dev','test'], [lemma_train, lemma_dev, lemma_test]):
            fname = join(reinflection_dir, f"{lang}.{pos}.lemma.{name}.txt")
            write_dataset(fname, subset, message=f"Writing the {name} file {fname}")


if __name__ == '__main__':
    main()
