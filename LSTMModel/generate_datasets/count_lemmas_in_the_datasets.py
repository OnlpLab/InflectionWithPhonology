from collections import defaultdict
from itertools import chain
from os.path import join
from typing import Callable, Collection, Dict, List, Tuple, Union

import pandas as pd
from pandas import DataFrame

from PhonologyConverter.g2p_config import sqi_clean_sample, lav_clean_sample, \
    hun_clean_sample, tur_clean_sample, fin_clean_sample

kat_clean_sample, swc_clean_sample, bul_clean_sample = [lambda x: x] * 3

cleaners = {'kat': kat_clean_sample, 'swc': swc_clean_sample, 'sqi': sqi_clean_sample, 'lav': lav_clean_sample,
            'bul': bul_clean_sample, 'hun': hun_clean_sample, 'tur': tur_clean_sample, 'fin': fin_clean_sample}


class LanguagePOSGroup:
    def __init__(self, name: str):
        self.name = name
        split_name = name.split('.')
        self.language = split_name[0]
        self.POS = split_name[1]


def parse_inflections_file(file_name: str, add_split_line_to_object: Callable,
                           empty_samples_object: Collection, line_typle='other', **kwargs) -> Collection:
    """
    Takes a file in inflection or reinflection format and parses every line to a samples object.
    :param file_name: the file to be processed
    :param add_split_line_to_object: a method that parses the line and aggregates it
    :param empty_samples_object: the initial object to be accumulated to, e.g. {}, []
    :param line_typle: if 'inflection' assumes the format {lemma}\t{form}\t{features}; if 'reinflection'
                 assumes {src_feat}\t{src_form}\t{trg_feat}\t{trg_form}
    :return: samples_object
    """
    samples_object = empty_samples_object
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            split_line = line.split("\t")
            if line_typle == 'inflection':
                lemma, form, features = split_line
                samples_object = add_split_line_to_object(samples_object, lemma, form, features, **kwargs)
            elif line_typle == 'reinflection':
                src_feat, src_form, trg_feat, trg_form = split_line
                samples_object = add_split_line_to_object(samples_object, src_feat, src_form, trg_feat, trg_form, **kwargs)
            else:
                samples_object = add_split_line_to_object(samples_object, split_line, **kwargs)
    return samples_object


# region accumulation_methods

def build_lemma_features_dict(samples_object: Dict, lemma, form, features) -> Dict:
    if lemma not in samples_object:
        samples_object[lemma] = {}
    samples_object[lemma][features] = form
    return samples_object


def build_form_features_dict(samples_object: defaultdict, lemma, form, features, **kwargs) -> Dict:
    language = kwargs['language']
    if language == 'hun':
        lemma = ''.join(cleaners[language](','.join(list(lemma))).split(','))
        form  = ''.join(cleaners[language](','.join(list(form))).split(','))
    else:
        lemma = cleaners[language](lemma)
        form  = cleaners[language](form)
    samples_object[(form, features)].append(lemma)
    return samples_object


def build_inflection_tuples(samples_object: List, lemma, form, features) -> List:
    samples_object.append((lemma, form, features))
    return samples_object


def build_reinflection_sample_dict(samples_object: List, src_feat, src_form, trg_feat, trg_form) -> List:
    samples_object.append({"src": (src_form, src_feat), "trg": (trg_form, trg_feat)})
    return samples_object


# endregion accumulation_methods

def read_reinflection_file(file_name: str) -> Union[List[Dict[str, Tuple[str, str]]], Collection]:
    return parse_inflections_file(file_name, build_reinflection_sample_dict, [], 'reinflection')


def count_lemmas_in_reinflection_file(paradigms, reinflection_samples: Union[List[Dict], Collection]) -> int:
    # For each pair, find the lemma(s) that match them. If there's more than 1, intersect the results for the src and trg forms.
    # Summarize the lemmas in a list, and return it and its size
    resulted_lemmas = set()
    for reinflection_sample in reinflection_samples:
        src_pair, trg_pair = reinflection_sample.values()
        src_lemmas, trg_lemmas = paradigms[src_pair], paradigms[trg_pair]
        possible_lemmas = list(set(src_lemmas) & set(trg_lemmas))
        assert len(possible_lemmas) >= 1
        if len(possible_lemmas) > 1:
            # print(f"found {len(possible_lemmas)} lemmas for the sample ({src_pair},{trg_pair}): {possible_lemmas}")
            # if any of the lemmas are already in resulted_lemmas, don't add anything else. Otherwise, add only the first.
            if not any(lemma in resulted_lemmas for lemma in possible_lemmas):
                # print(f"Inserting the new lemma {possible_lemmas[0]} for the sample ({src_pair},{trg_pair})")
                resulted_lemmas.add(possible_lemmas[0])
        else:
            resulted_lemmas.add(possible_lemmas[0])

    return len(resulted_lemmas)

def write_to_excel(excel_outputs_file, inflection_stats_list, reinflection_stats_list):
    inflection_df = DataFrame(inflection_stats_list, columns=['Language', 'POS', '#Lemmas', '#Forms', '#Forms/#Lemmas']).fillna("")
    reinflection_df = DataFrame(reinflection_stats_list, columns=['Language', 'POS', 'Split', 'FileType', '#Lemmas', '#Forms', '#Forms/#Lemmas']).fillna("")

    with pd.ExcelWriter(excel_outputs_file) as writer:
        inflection_df.to_excel(writer, sheet_name='Inflection')
        reinflection_df.to_excel(writer, sheet_name='Reinflection')


def main():
    file_types_sizes, splits = {'train': 8000, 'dev': 1000, 'test': 1000}, ['form', 'lemma']
    lang_pos_list = ['bul.ADJ', 'bul.V', 'fin.ADJ', 'fin.N', 'fin.V', 'hun.V', 'kat.N', 'kat.V',
                     'lav.N', 'lav.V', 'sqi.V', 'swc.ADJ', 'swc.V', 'tur.ADJ', 'tur.V']
    lang_pos_groups = [LanguagePOSGroup(group) for group in lang_pos_list]
    inflection_files_folder = join('..', '..', '.data', 'InflectionTables')
    reinflection_files_folder = join('..', '..', '.data', 'Reinflection', 'CleanedData')

    excel_outputs_file = "Data-Stats.xlsx"
    inflection_stats_list, reinflection_stats_list = [], []

    for lang_pos_group in lang_pos_groups:
        inflection_file_name = join(inflection_files_folder, f'{lang_pos_group.name}.txt')
        print(f"{lang_pos_group.name}:")

        inflection_items = dict(parse_inflections_file(inflection_file_name, build_form_features_dict, defaultdict(list),
                                                       'inflection', language=lang_pos_group.language))

        lemmas_lists_instances = [v for v in inflection_items.values()]
        total_lemmas_number = len(set(chain(*lemmas_lists_instances)))
        total_forms_number = sum([len(v) for v in lemmas_lists_instances])
        avg_paradigm_size = round(total_forms_number / total_lemmas_number, 1)
        print(f"Raw data: #lemmas = {total_lemmas_number}, #forms = {total_forms_number}. avg-table-size = {avg_paradigm_size}")
        inflection_stats_list.append([lang_pos_group.language, lang_pos_group.POS, total_lemmas_number, total_forms_number, avg_paradigm_size])

        for split in splits:
            print(f"{split} split:")
            for file_type in file_types_sizes.keys():
                reinflection_file_name = join(reinflection_files_folder, lang_pos_group.name,
                                              f'{lang_pos_group.name}.{split}.{file_type}.txt')

                reinflection_samples = read_reinflection_file(reinflection_file_name)
                assert len(reinflection_samples) == file_types_sizes[file_type]

                lemmas_number = count_lemmas_in_reinflection_file(inflection_items, reinflection_samples)
                forms_number = file_types_sizes[file_type] * 2
                forms_lemmas_ratio = round(forms_number / lemmas_number, 2)
                print(f"\tReinflection {file_type}: #lemmas = {lemmas_number}, #forms = {forms_number}, avg-ratio = {forms_lemmas_ratio}")
                reinflection_stats_list.append([lang_pos_group.language, lang_pos_group.POS, split, file_type, lemmas_number, forms_number, forms_lemmas_ratio])
            print()

    write_to_excel(excel_outputs_file, inflection_stats_list, reinflection_stats_list)

if __name__ == '__main__':
    main()
