import os
from os.path import join, basename, isfile
from argparse import ArgumentParser

REINFLECTION_STR, INFLECTION_STR = 'reinflection', 'inflection'
def reinflection2sample(line, mode=REINFLECTION_STR):
    assert mode in {REINFLECTION_STR, INFLECTION_STR}
    if mode==REINFLECTION_STR:
        # The line format is [src_feat, src_form, trg_feat, trg_form]
        src_feat, src_form, trg_feat, trg_form = line
        src_feat, trg_feat = src_feat.split(";"), trg_feat.split(";")
        src_form, trg_form = list(src_form), list(trg_form)
        src = ','.join(src_feat + ['+'] + src_form + ['+'] + trg_feat)
        trg = ','.join(trg_form)
    else: # inflection mode
        lemma, form, feat = line
        feat = feat.split(";")
        lemma, form = list(lemma), list(form)
        src = ','.join(lemma + ['$'] + feat) # Don't use '+' as it is part of some tags (see the file tags.yaml).
        trg = ','.join(form)
    return src, trg

def reinflection2TSV(fn, new_fn, mode=REINFLECTION_STR):
    """
    Convert a file in the Reinflection format (src_feat\tsrc_form\ttrg_feat\ttrg_form) to a TSV file of the format
    src\ttrg, each one consists of CSV strings. If mode=inflection, then count the data as SIGMORPHON format, and fn must be a tuple of 3 paths.
    :param fn: if mode=reinflection, then fn: str. else, fn:Tuple(str)
    :param new_fn:
    :param mode: can be either 'inflection' or 'reinflection'.
    :return: The two paths of the TSV files.
    """
    assert mode in {REINFLECTION_STR, INFLECTION_STR}
    if mode==REINFLECTION_STR:
        data = open(fn, encoding='utf8').readlines()
        data = [line.strip().split('\t') for line in data]

        examples = []
        for e in data:
            if e[0] == '': continue
            src, trg = reinflection2sample(e, mode=mode) # the only modification for supporting Inflection as well.
            examples.append(f"{src}\t{trg}")

        open(new_fn, mode='w', encoding='utf8').write('\n'.join(examples))
    else:
        train_fn, dev_fn, test_fn = fn # file paths without parent-directories prefix
        new_train_fn = basename(train_fn)+".tsv" # use the paths without parent-directories prefixes
        new_dev_fn = basename(dev_fn)+".tsv"
        new_test_fn = basename(test_fn)+".tsv"
        if isfile(new_train_fn) and isfile(new_dev_fn) and isfile(new_test_fn): # don't do it again
            return [new_train_fn, new_dev_fn, new_test_fn]

        for fn,new_fn in zip([train_fn, dev_fn, test_fn], [new_train_fn, new_dev_fn, new_test_fn]):
            data = open(fn, encoding='utf8').read().split('\n')
            data = [line.split('\t') for line in data]

            examples = []
            for e in data:
                if e[0] == '': continue
                src, trg = reinflection2sample(e, mode=mode) # the only modification for supporting Inflection as well.
                examples.append(f"{src}\t{trg}")

            open(new_fn, mode='w', encoding='utf8').write('\n'.join(examples))
        new_fn = [new_train_fn, new_dev_fn, new_test_fn]
        # The result is a directory "LEMMA_TSV_FORMAT" with 180 files of the format 'lang.{trn|tst}.tsv'
    return new_fn

def main():
    parser = ArgumentParser(description="Generate TSV files from the Reinflection files")
    parser.add_argument("lang", type=str, help="Language's files to be processed")
    parser.add_argument("POS", nargs='?', type=str, choices=['V','N','ADJ'], help="Part of speech's files to be processed")
    parser.add_argument('training_mode', nargs='?', type=str, choices=['form', 'lemma', 'both'], help="Can be either form-split, lemma-split or both")
    args = parser.parse_args()
    lang, POS, training_mode = args.lang, args.POS, args.training_mode
    if POS is None and training_mode is None: assert lang=='all'

    if lang=='all':
        data_dir = join("../.data", "Reinflection")
        for sub_dir in os.listdir(data_dir):
            flag = False
            subsub_dir = join(data_dir, sub_dir)
            for path in os.listdir(subsub_dir):
                path_wo_ext, ext = os.path.splitext(path)
                new_f = join(subsub_dir, path_wo_ext+'.tsv')
                if ext=='.txt' and not os.path.isfile(new_f):
                        reinflection2TSV(join(subsub_dir, path), new_f)
                        flag=True
            if flag: print(f"Converted files in {sub_dir} to .tsv format.")
            else: print(f"No files were converted in {sub_dir}!")
                # assert ext=='.txt', f"the extension of the file {join(subsub_dir, path)} must be .txt !"
    else:
        data_dir = join("../.data", "Reinflection", f"{lang}.{POS}")
        if training_mode=='both':
            for tm in ['form', 'lemma']:
                for a in ['train', 'dev', 'test']:
                    path = join(data_dir, f"{lang}.{POS}.{tm}.{a}.txt")
                    new_path = join(data_dir,f"{lang}.{POS}.{tm}.{a}.tsv")
                    reinflection2TSV(path, new_path)
        else:
            for a in ['train', 'dev', 'test']:
                path = join(data_dir, f"{lang}.{POS}.{training_mode}.{a}.txt")
                new_path = join(data_dir,f"{lang}.{POS}.{training_mode}.{a}.tsv")
                reinflection2TSV(path, new_path)

if __name__ == '__main__':
    main()