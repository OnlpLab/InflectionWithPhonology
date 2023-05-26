from os.path import join, isdir, basename
from os import mkdir
import random
from data2samples_converter import Data2SamplesConverter
from copy import deepcopy
from tqdm import tqdm
import codecs
from argparse import ArgumentParser
random.seed(1)

# for debugging purposes:
def find_common_features(lemmas:[dict]) -> [str]:
    features_list = [set(lemma.keys()) for lemma in lemmas] # generalization of feats1, feats2 = set(lemma1.keys()), set(lemma2.keys())
    return set.intersection(*features_list)

def has_features(paradigm_entries: dict, features:[str]) -> bool:
    return set(features).issubset(paradigm_entries.keys())

def filter_dict_by_features(lemmas:dict, features:[str]) -> dict:
    """
    Filter lemmas_dict (supposed to be a dict composed of train/dev/test reinflection samples) by given features.
    """
    return {lemma_name:feature_form_entries for lemma_name, feature_form_entries in lemmas.items()
            if has_features(feature_form_entries, features)}

def join_fo(w:str): return w.replace(',','')
def join_fe(w:str): return w.replace(',',';')
def spl_fe(w:str): return w.split(';')
def spl_fo(w:str): return list(w)

def read_inflection_tables(inflection_file):
    """ read a standard inflection file (of the format "lemma\tform\tfeature") and construct a standard dictionary:
     {lemma1: {feature1: form1, feature2: form2, ...}, lemma2: {feature1: form1, feature2: form2, ...}, ... } """
    D = {}
    with codecs.open(inflection_file, 'rb', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lemma, word, tag = line.split("\t")
            if lemma not in D:
                D[lemma] = {}
            D[lemma][tag] = word
    return D

def read_reinflection_samples(file_name):
    """ Read a reinflectin-format file """
    data = []
    with open(file_name, encoding='utf8') as f:
        for line in f:
            elements = line.strip('\n').split('\t') # src_feat, src_form, trg_feat, trg_form
            assert len(elements) == 4
            data.append(tuple(elements))
    return data

def inflection_paradigms2list_and_dict(inflection_file):
    """
    :param inflection_file: a standard Inflection file.
    :return: a dictionary of the form { feature1: {form1:lemma1, form2:lemma2, ...},
     feature2:{form1:lemma1, form2:lemma2, ...}, ... } , and also a list of (lemma, form, feature) tuples
    """
    inflection_paradigms = read_inflection_tables(inflection_file)
    inflection_tuples = []
    features_occurences = {}
    for lemma, feature_form_entries in inflection_paradigms.items():
        for feature, form in feature_form_entries.items():
            inflection_tuples.append((lemma, form, feature))
            if feature not in features_occurences:
                features_occurences[feature] = {}
            features_occurences[feature][form] = lemma
    return features_occurences, inflection_tuples

def track_kat_V_lemma(features_occurences, inflections_tuples, src_feat, src_form, trg_feat, trg_form):
    if 'MSDR' in src_feat and 'MSDR' not in trg_feat:
        lemma = features_occurences[trg_feat][trg_form] # since identical Masdar forms might occur at both
        # the active & passive Gerund forms of a root, we use the other form to distinguish between them.
    elif 'MSDR' in src_feat and 'MSDR' in trg_feat: # we're dealing with both Perfective & Imperfective gerunds of the same verb.
        # Find one tuple where both one of the forms appear, and just attach it the tuple's lemma
        # (it's necessarily a solution, even if not the single one).
        lemma = [t for t in inflections_tuples if src_form in t][0][0]
    else:
        # if the source is not Masdar, we know exactly its lemma
        lemma = features_occurences[src_feat][src_form] # == features_dictionary[trg_feat][trg_form]
    return lemma

def group_reinflection_samples_by_lemmas(features_occurences, inflections_tuples, reinflection_file):
    """
    :param features_occurences: as described at inflection_paradigms2list_and_dict
    :param inflections_tuples: as described at inflection_paradigms2list_and_dict
    :param reinflection_file: a file of reinflection samples
    :return: a sub-dictionary of the one generated when applying read_inflection_tables to the corresponding inflection file
    """
    reinflection_samples = read_reinflection_samples(reinflection_file)
    inflections_subdictionary = {}
    for reinflection_sample in reinflection_samples:
        src_feat, src_form, trg_feat, trg_form = reinflection_sample
        if lang == 'kat' and POS == 'V':
            lemma = track_kat_V_lemma(features_occurences, inflections_tuples, src_feat, src_form, trg_feat, trg_form)
        else:
            lemma = features_occurences[src_feat][src_form]

        if lemma not in inflections_subdictionary:
            inflections_subdictionary[lemma] = {}
        inflections_subdictionary[lemma][src_feat] = src_form
        inflections_subdictionary[lemma][trg_feat] = trg_form
    return inflections_subdictionary


class AnalogyFunctionality(Data2SamplesConverter):
    """
    A class for generating new train-dev-test datasets based on Analogy methods. The datasets consist of the original
    datasets samples, combined with "analogised" data. The datasets proportions is pre-defined by those of the
    original datasets.

    Note: there's no difference between the form-split and the lemma-split implementations, because this separation
    is only done between the original train-dev-test samples. In fact, the auxiliary samples used for
    analogising can be used in both the train and test sets!
    """
    def __init__(self, new_files_dir, inflections_file, train_file, dev_file, test_file):
        """
        :param new_files_dir: the location of the new analogy files to be generated
        :param inflections_file: the complete list of paradigms from which the reinflection samples were generated
        :param train_file: the name of the train reinflection samples file.
        :param dev_file: the name of the dev reinflection samples file.
        :param test_file: the name of the test reinflection samples file.
        """
        super().__init__()

        self.inflection_data_file = inflections_file
        self.all_tables_by_lemmas = read_inflection_tables(self.inflection_data_file)
        self.new_dir = new_files_dir
        self.train_file, self.dev_file, self.test_file = train_file, dev_file, test_file

        # store the reinflection samples of each set:
        self.train_samples = read_reinflection_samples(self.train_file)
        self.dev_samples = read_reinflection_samples(self.dev_file)
        self.test_samples = read_reinflection_samples(self.test_file)

        self.features_dictionary, self.inflections_tuples = inflection_paradigms2list_and_dict(inflections_file)
        self.train_lemmas_dictionary = group_reinflection_samples_by_lemmas(self.features_dictionary, self.inflections_tuples, self.train_file)

    def generate_cross_1_datasets(self):
        """
        Either in form-split or lemma-split mode, the chosen auxiliary lemmas are taken from the train set. The possible lemmas are chosen by the current features.
        The goal is to filter the train lemmas according to the specific src & trg features (that are taken from the dev/test sets -- but it doesn't matter).
        :return: 3 lists of src1-cross1 reinflection samples
        """
        train_samples = deepcopy(self.train_samples)
        dev_samples = deepcopy(self.dev_samples)
        test_samples = deepcopy(self.test_samples)
        orignal_samples_lists = [train_samples, dev_samples, test_samples]
        new_samples = [] # going to be a list of tuples
        train_lemmas_dictionary = deepcopy(self.train_lemmas_dictionary) # a copy of the partial tables from the train set

        for orignal_samples_list in orignal_samples_lists:
            new_samples.append([])
            for reinflection_sample in tqdm(orignal_samples_list):
                src_feat, src_form, trg_feat, trg_form = reinflection_sample

                # find out the lemma from which src_form & trg_form were inflected
                if lang == 'kat' and POS == 'V':
                    lemma = track_kat_V_lemma(self.features_dictionary, self.inflections_tuples, src_feat, src_form, trg_feat, trg_form)
                else:
                    lemma = self.features_dictionary[src_feat][src_form] # == features_dictionary[trg_feat][trg_form]

                # Choose a different lemma, that includes the entries src_feat and trg_feat in the train set.
                possible_lemmas_dict = filter_dict_by_features(train_lemmas_dictionary, [src_feat, trg_feat])
                possible_lemmas = list(possible_lemmas_dict.keys())
                # If needed, make sure the same lemma isn't chosen again
                if lemma in possible_lemmas:
                    possible_lemmas.remove(lemma)

                # If there are no lemmas that follow the criterion above, select a random train-lemma. This could
                # happen when working on the dev/test sets
                if len(possible_lemmas) == 0:
                    possible_lemmas_dict = filter_dict_by_features(self.all_tables_by_lemmas, [src_feat, trg_feat])
                    possible_lemmas = list(possible_lemmas_dict.keys())
                    if lemma in possible_lemmas:
                        possible_lemmas.remove(lemma)

                chosen_lemma = random.sample(possible_lemmas, k=1)[0]
                # Once chose the lemma, take the forms of these features
                aux_src_form = possible_lemmas_dict[chosen_lemma][src_feat]
                aux_trg_form = possible_lemmas_dict[chosen_lemma][trg_feat]
                # Arrange all the elements into the new sample according to an agreed schema.
                new_sample = ((src_feat, src_form, aux_src_form, aux_trg_form, trg_feat), trg_form)

                new_samples[-1].append(new_sample)
        return new_samples

    @staticmethod
    def _cross1_reinflection(line: ([str], str)) ->  ([[str]], [str]):
        (src_feat, src_form, aux_src_form, aux_trg_form, trg_feat), trg_form = line
        return [spl_fe(src_feat), spl_fo(src_form), spl_fo(aux_src_form), spl_fo(aux_trg_form), spl_fe(trg_feat)], spl_fo(trg_form)

    @staticmethod
    def _cross1_sample2data(src:str, trg: str) -> ([[str]], str):
        fe1, fo1, fo2, fo3, fe2 = src.split(',+,')
        fe1, fo1, fo2, fo3, fe2 = join_fe(fe1), join_fo(fo1), join_fo(fo2), join_fo(fo3), join_fe(fe2)
        src = fe1, fo1, fo2, fo3, fe2
        trg = ''.join(trg.split(','))
        return src, trg

    def analogy_reinflection2TSV(self, original_dir, fn, data) -> str:
        # Encapsulating the automatic invoking of the parent method, but with different suffix and parsing method.
        return self.reinflection2TSV(fn, suffix='.src1_cross1', old_dir=original_dir, new_dir=self.new_dir, parsing_func=self._cross1_reinflection, data=data)

    def analogy_sample2data(self, inp, has_sources=False):
        # limited tells whether the source sequences are supplied.
        if has_sources: # no sources
            src, trg, pred = inp
            pred = ''.join(pred)
            src, trg = self.sample2reinflection((src, trg), parsing_func=self._cross1_sample2data)
            return src, trg, pred
        else:
            trg, pred = inp
            return ''.join(trg), ''.join(pred)


def main():
    original_dir = join("../.data", "Reinflection", f"{lang}.{POS}")
    analogies_dir = join(original_dir, analogy_type)
    inflections_file = join("../.data", "InflectionTables", f"{lang}.{POS}.txt")
    if not isdir(analogies_dir): mkdir(analogies_dir)

    print(f"Generating {analogy_type}-analogy datasets for {lang}.{POS}:")
    for training_mode in ['form', 'lemma']:
        original_files = [join(original_dir, f"{lang}.{POS}.{training_mode}.{e}.txt") for e in ['train', 'dev', 'test']]

        kat_analogies = AnalogyFunctionality(analogies_dir, inflections_file, *original_files)
        analogised_train_dev_test_data = kat_analogies.generate_cross_1_datasets() # analogised_train_dev_test_data is a [train, dev, test] list of datasets
        print(f"Generating {training_mode}-split {analogy_type} analogy datasets:")

        for original_file, new_analogies_data in zip(original_files, analogised_train_dev_test_data): # iterates thrice - train, dev, test
            new_analogy_file = kat_analogies.analogy_reinflection2TSV(original_dir, basename(original_file), data=new_analogies_data)
            print(f"Generated data for {new_analogy_file}")

def validate_kat_V_dataset(training_mode, subset):
    # A short script for kat.V, that makes sure the chosen auxiliary lemmas are differet from the original ones.
    analogy_type = 'src1_cross1'
    original_dir = join("../.data", "Reinflection", f"{lang}.{POS}")
    analogies_dir = join(original_dir, analogy_type)
    inflections_file = join("../.data", "InflectionTables", f"{lang}.{POS}.txt")
    original_files = [join(original_dir, f"{lang}.{POS}.{training_mode}.{e}.txt") for e in ['train', 'dev', 'test']]

    kat_analogies = AnalogyFunctionality(analogies_dir, inflections_file, *original_files)

    train_analog_path = join(analogies_dir, f"{lang}.{POS}.{training_mode}.{subset}.{analogy_type}.tsv")
    train_lines = open(train_analog_path, encoding='utf8').read().split('\n')
    train_lines = [line.strip().split('\t') for line in train_lines]
    train_data = []
    i=0
    for line in train_lines:
        src, trg = kat_analogies.analogy_sample2data(line)
        src, trg = kat_analogies._cross1_sample2data(src, trg)
        fe1, fo1, fo_aux1, fo_aux2, fe2 = src
        feature1_entry, feature2_entry = kat_analogies.features_dictionary[fe1], kat_analogies.features_dictionary[fe2]
        bads = [] # 1st and 2nd error types stand for ambiguities at the lemmas correspond to a given Masdar forms.
        if not feature1_entry[fo1] == feature2_entry[trg]:
            bads.append((1, fo1, trg)) # Don't worry if it's printed
        if not feature1_entry[fo_aux1] == feature2_entry[fo_aux2]:
            bads.append((2, fo_aux1, fo_aux2)) # Don't worry if it's printed
        if feature1_entry[fo1]==feature1_entry[fo_aux1]:
            bads.append((3, fo1, fo_aux1)) # Do worry if it's printed
        if bads:
            print(f"{i+1}. The sample: {src, trg}. Bad ones: {bads}")
            i+=1
        train_data.append((src,trg))
    print("Done")

if __name__ == '__main__':
    # Note: this file was not run on tur-N and swc-N, because they're not used later. Also, swc-N can't generate lemma-split.
    parser = ArgumentParser(description="Parse arguments for generating Analogy src1-cross1 datasets")
    parser.add_argument('lang', type=str, choices=['bul', 'fin', 'hun', 'kat', 'lav', 'sqi', 'swc', 'tur'], help="Language to be processed")
    parser.add_argument('POS', type=str, choices=['V','N','ADJ'], help="Part of speech to be processed")

    args = parser.parse_args()
    lang, POS = args.lang, args.POS
    analogy_type = 'src1_cross1'
    main()

    # region validate kat.V
    if lang == 'kat' and POS == 'V':
        for a in ['form', 'lemma']:
            for b in ['train', 'dev', 'test']:
                validate_kat_V_dataset(a, b)
    # endregion validate kat.V
