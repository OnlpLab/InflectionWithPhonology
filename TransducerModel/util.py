import codecs
from os.path import join, isfile, dirname
from os import listdir
from ast import literal_eval
from typing import Tuple, Dict
from editdistance import eval as edit_distance_eval

from defaults import EVALM_PATH, LANGUAGES_LIST
from PhonologyConverter.languages_setup import LanguageSetup

def are_substrings_in_string(target_string: str, substrings: tuple) -> bool:
    return all([substring in target_string for substring in substrings])

def get_language(file_name: str) -> str:
    """
    Takes a file name (one of the train / dev / test files, and finds the run's language,
    given the languages list.
    """
    query = [are_substrings_in_string(file_name, (lang, )) for lang in LANGUAGES_LIST] # returns a "True-hot" list
    assert query.count(True) == 1
    return LANGUAGES_LIST[query.index(True)]


def write_stats_file(dev_accuracy, paths, data_arguments, model_arguments, optim_arguments):

    with open(paths['stats_file_path'], 'w') as w:
        
        print('LANGUAGE: {}, REGIME: {}'.format(paths['lang'], paths['regime']), file=w)
        print('Train path:   {}'.format(paths['train_path']), file=w)
        print('Dev path:     {}'.format(paths['dev_path']), file=w)
        print('Test path:    {}'.format(paths['test_path']), file=w)
        print('Results path: {}'.format(paths['results_file_path']), file=w)
        
        for k, v in paths.items():
            if k not in ('lang', 'regime', 'train_path', 'dev_path',
                         'test_path', 'results_file_path'):
                print('{:20} = {}'.format(k, v), file=w)
        print(file=w)
        
        for name, args in (('DATA ARGS:', data_arguments),
                           ('MODEL ARGS:', model_arguments),
                           ('OPTIMIZATION ARGS:', optim_arguments)):
            print(name, file=w)
            for k, v in args.items():
                print('{:20} = {}'.format(k, v), file=w)
            print(file=w)
       
        print('DEV ACCURACY (internal evaluation) = {}'.format(dev_accuracy), file=w)


def external_eval(output_path, gold_path, batches, predictions, sigm2017format, evalm_path=EVALM_PATH):
    pred_path = output_path + 'predictions'
    line = '{IFET}\t{IN}\t{FET}\t{WORD}\t{GOLD}\n'

    # WRITE FILE WITH PREDICTIONS
    with codecs.open(pred_path, 'w', encoding='utf8') as w:
        for sample, prediction in zip((s for b in batches for s in b), predictions):
            w.write(line.format(IFET=sample.in_feat_str, IN=sample.lemma_str, FET=sample.out_feat_str, WORD=prediction, GOLD=sample.word_str))


def write_generalized_measures(stats_file, measures: Tuple[int, int, int, int]):
    graphemes_accuracy, features_accuracy, graphemes_ed, features_ed = measures
    with open(stats_file, 'a+') as f:
        f.write(f"Evaluating based on the predictions file:\n")
        f.write(f"Features-level: Accuracy: {features_accuracy}, Edit Distance: {features_ed}\n")
        f.write(f"Graphemes-level: Accuracy: {graphemes_accuracy}, Edit Distance: {graphemes_ed}\n")

def evaluate_pred_vs_gold(features_prediction: Tuple[str], graphemes_gold: str, phonology_converter: LanguageSetup, output_mode: str) -> Dict:
    graphemes_prediction = phonology_converter.phonemes2word(features_prediction, output_mode, normalize=True)
    features_gold = tuple(phonology_converter.word2phonemes(graphemes_gold, output_mode))

    return {'graphemes_equality': graphemes_gold == graphemes_prediction,
            'graphemes_ed': edit_distance_eval(graphemes_gold, graphemes_prediction),
            'features_equality': features_gold == features_prediction,
            'features_ed': edit_distance_eval(features_gold, features_prediction)}


def evaluate_features_predictions(outputs_file: str, phonology_converter: LanguageSetup = None, output_mode='features'):
    """
    Takes a file of the output format '{IFET}\t{IN}\t{FET}\t{WORD}\t{GOLD}', reads the last 2 in each line and calculates 4 measures.
    Note: this method should only be called if reevaluation is required, i.e. the output format is features/phonemes.
    """
    try:
        rows = open(outputs_file, encoding='utf8').readlines()
        pairs = [line.strip().split('\t')[-2:] for line in rows]
    except FileNotFoundError:
        print(f"Warning: No such file: {outputs_file}")
        return

    # print(join(dirname(outputs_file), 'f.stats'))
    assert isfile(join(dirname(outputs_file), 'f.stats'))

    if phonology_converter is None:
        language = get_language(outputs_file)
        phonology_converter = LanguageSetup.create_phonology_converter(language)

    measures_per_pair = [evaluate_pred_vs_gold(literal_eval(pair[0]), pair[1], phonology_converter, output_mode) for pair in pairs]

    average_measure_by_key = lambda key: sum([sample_results_dict[key] for sample_results_dict in measures_per_pair]) / len(pairs)
    graphemes_accuracy = average_measure_by_key('graphemes_equality')
    features_accuracy = average_measure_by_key('features_equality')
    graphemes_ed = average_measure_by_key('graphemes_ed')
    features_ed = average_measure_by_key('features_ed')

    return graphemes_accuracy, features_accuracy, graphemes_ed, features_ed

if __name__ == '__main__':
    single_file, local_mode = False, False
    if single_file:
        print(evaluate_features_predictions(join("Results", "Outputs1007__bul_V_lemma_f_f_None_42", 'f.greedy.test.predictions'), output_mode='phonemes'))
    elif local_mode:
        results_folder = join('.', 'Results')
        not_features_runs = ['Outputs3__kat_V_form_g_g_None_42', 'Outputs4__kat_V_form_g_g_None_42', 'Outputs__kat_V_form_g_g_None_42']
        predictions_files = [join(results_folder, f, 'f.greedy.test.predictions') for f in listdir(results_folder) if f not in not_features_runs]
        files_to_iterate = list(set(predictions_files) - set(not_features_runs))

        for pred_file in files_to_iterate:
            print(f"{pred_file}: ", end='')
            print(evaluate_features_predictions(pred_file))
    else:
        results_folder = join('.', 'Results', 'test_runs', 'f-p-attn')
        predictions_files = [join(results_folder, f, 'f.greedy.test.predictions') for f in listdir(results_folder)]

        for pred_file in predictions_files:
            print(f"{pred_file}: ", end='')
            print(evaluate_features_predictions(pred_file, output_mode='phonemes'))