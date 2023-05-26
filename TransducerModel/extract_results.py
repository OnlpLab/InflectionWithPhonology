# This is a Python3 script for extracting the runs results to an Excel file, including final dev and test accuracies.
import re
from itertools import product
from datetime import datetime as dt
from os import listdir
from os.path import basename, isdir, join
from pandas import DataFrame
from util import are_substrings_in_string

CUTOFF = len("Outputs_2023-01-28 135937_")

def find_float_in_line(line):
    matches = re.findall("\d+\.\d+", line)
    assert len(matches) == 1, "Invalid output!"
    return float(matches[0])

def find_2_floats_in_line(line):
    a, b = re.findall("\d+\.\d+", line)
    return float(a), float(b)

def extract_accuracies_from_single_folder(folder, io_format):
    # The folder has the format: Outputs_YYYY-MM-DD hhmmss_lang_pos_training-mode_IO_analogy_seed.
    # If the output format is not graphemes ('g'), then reevaluation was made at graphemes level.

    try:
        lines = open(join(folder, 'f.stats'), encoding='utf8').read().split('\n')
    except FileNotFoundError:
        print(f"No f.stats file in {folder}!")
        return None

    test_ed, test_graphemes_accuracy, test_graphemes_ed = [-1] * 3
    if io_format == 'gg':
        dev_accuracy, test_accuracy = find_float_in_line(lines[-3]), find_float_in_line(lines[-2])

    else:  # io_format in {'ff', 'fp'}
        # See util.write_generalized_measures; dev_ and test_ accuracy mean here eval at features level
        dev_accuracy, test_accuracy = find_float_in_line(lines[-6]), find_float_in_line(lines[-5])

        test_features_accuracy, test_ed = find_2_floats_in_line(lines[-3])
        assert test_features_accuracy == test_accuracy

        test_graphemes_accuracy, test_graphemes_ed = find_2_floats_in_line(lines[-2])

    return dev_accuracy, test_accuracy, test_ed, test_graphemes_accuracy, test_graphemes_ed


def main():
    results_folder = join('.', 'Results', 'LearningCurves')
    io_formats = ['gg', 'ff', 'fp']
    analogy_types, seeds = ['None'], ['100', '42', '21']
    train_samples = ['1000', '2000', '3000', '4000', '5000', '6000', '7000']

    excel_results_file = f"Test-Results {'_'.join(analogy_types)}-{'_'.join(seeds)}-{'_'.join(io_formats)}-1k-7k - {dt.now().strftime('%d%m%Y_%H%M%S')}.xlsx"

    run_records = []
    for seed in seeds:
        parent_folder = join(results_folder, f"{seed}")  # , f"{analogy}_{seed}_{io_format}")
        assert isdir(parent_folder), f"Folder {parent_folder} doesn't exist!"

        folders = []
        for f in listdir(parent_folder):

            for current_combination in product(analogy_types, io_formats, train_samples):
                if isdir(join(parent_folder, f)) and are_substrings_in_string(f[CUTOFF:], current_combination):
                    folders.append(join(parent_folder, f))

        for folder in folders:
            datetime, lang, pos, training_mode, io_format, analogy, _, samples_number = basename(folder).split("_")[1:]
            metrics = extract_accuracies_from_single_folder(folder, io_format)
            if metrics is None: continue

            run_records.append([datetime, analogy, seed, lang, pos, training_mode,
                                io_format, samples_number, *metrics])

    df = DataFrame(run_records, columns=['DateTime', 'AnalogyMode', 'Seed', 'Language', 'POS', 'Form/Lemma', 'IO mode',
                                         '#Train Samples', 'Dev Accuracy', 'Test Accuracy', 'Test ED',
                                         'Test Graphemes Accuracy', 'Test Graphemes ED'])
    df = df.fillna("")
    df.to_excel(excel_results_file)


if __name__ == '__main__':
    main()
