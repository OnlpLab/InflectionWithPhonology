"""Trains and evaluates a state-transition model for inflection generation, using the sigmorphon 2017 shared task
data files and evaluation script.

Usage:
  run_transducer.py [--dynet-seed SEED] [--dynet-mem MEM] [--dynet-autobatch ON]
  [--transducer=TRANSDUCER] [--sigm2017format] [--no-feat-format] [--use-phonology] [--self-attn] [--train-samples=T]
  [--input=INPUT] [--feat-input=FEAT] [--action-input=ACTION] [--pos-emb] [--avm-feat-format]
  [--enc-hidden=HIDDEN] [--dec-hidden=HIDDEN] [--enc-layers=LAYERS] [--dec-layers=LAYERS]
  [--vanilla-lstm] [--mlp=MLP] [--nonlin=NONLIN] [--lucky-w=W]
  [--pretrain-dropout=DROPOUT] [--dropout=DROPOUT] [--l2=L2]
  [--optimization=OPTIMIZATION] [--batch-size=BATCH-SIZE] [--decbatch-size=BATCH-SIZE]
  [--patience=PATIENCE] [--epochs=EPOCHS] [--pick-loss]
  [--align-smart | --align-dumb | --align-cls] [--tag-wraps=WRAPS] [--try-reverse | --iterations=ITERATIONS]
  [--substitution | --copy-as-substitution] [--param-tying]
  [--mode=MODE] [--verbose] [--beam-width=WIDTH] [--beam-widths=WIDTHS]
  [--pretrain-epochs=EPOCHS | --pretrain-until=ACC] [--sample-size=SAMPLE-SIZE] [--scale-negative=S]
  [--alpha=ALPHA] [--beta=BETA] [--no-baseline]
  TRAIN-PATH DEV-PATH RESULTS-PATH [--test-path=TEST-PATH] [--reload-path=RELOAD-PATH] [--hall-path=HALL-PATH]

Arguments:
  TRAIN-PATH    destination path, possibly relative to "data/all/", e.g. task1/albanian-train-low
  DEV-PATH      development set path, possibly relative to "data/all/"
  RESULTS-PATH  results file to be written, possibly relative to "results"

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET
  --dynet-autobatch ON          perform automatic minibatching
  --transducer=TRANSDUCER       transducer model to use: hacm / st-haem / haem / hard [default: haem]
  --sigm2017format              assume sigmorphon 2017 input format (lemma, word, feats)
  --no-feat-format              no features format (input, *, output)
  --input=INPUT                 character embedding dimension [default: 100]
  --feat-input=FEAT             feature embedding dimension.  "0" denotes "bag-of-features". [default: 20]
  --action-input=ACTION         action embedding dimension [default: 100]
  --pos-emb                     embedding POS (or the first feature in the sequence of features) as a non-atomic feature
  --avm-feat-format             features are treated as an attribute-value matrix (`=` pairs attributes with values)

  --train-samples=T               number of train samples. Used for the learning curves. [default: 8000]
  --use-phonology               instead of transducing letters, use their phonological representeation
  --self-attn                   if use-phonology, then before the encoder rnn, represent phonemes by features with a self-attention layer

  --enc-hidden=HIDDEN           hidden layer dimension of encoder RNNs [default: 200]
  --enc-layers=LAYERS           number of layers in encoder RNNs [default: 1]
  --dec-hidden=HIDDEN           hidden layer dimension of decoder RNNs [default: 200]
  --dec-layers=LAYERS           number of layers in decoder RNNs [default: 1]
  --vanilla-lstm                use vanilla LSTM instead of DyNet 1's default coupled LSTM
  --mlp=MLP                     MLP hidden layer dimension. "0" denotes "no hidden layer". [default: 0]
  --nonlin=NONLIN               if mlp, this non-linearity is applied after the hidden layer. ReLU/tanh [default: ReLU]
  --lucky-w=W                   if feat-input==0, scale the "bag-of-features" vector by W [default: 55]
  --dropout=DROPOUT             variotional dropout in decoder RNN [default: 0.5]
  --pretrain-dropout=DROPOUT    if pretraining with MLE, this dropout rate is used, otherwise the rate of '--dropout'
  --optimization=OPTIMIZATION   optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: ADADELTA]
  --l2=L2                       l2-regularization coefficient. Regularization is applied to RNNs and classifier. [default: 0]
  --batch-size=BATCH-SIZE       batch size [default: 1]
  --decbatch-size=BATCH-SIZE    batch size for decoding [default: 1]
  --patience=PATIENCE           maximal patience for early stopping [default: 10]
  --epochs=EPOCHS               number of training epochs [default: 30]
  --pick-loss                   best model should have the highest dev loss (and not dev accuracy)
  --align-smart                 align with Chinese restaurant process like in Aharoni & Goldberg paper. Default.
  --align-dumb                  align by padding the shortest string on the right (lemma or inflected word)
  --align-cls                   align by aligning the strings' common longest substring first and then padding both strings.
  --try-reverse                 if align-dumb, try reversing lemma and word strings if no COPY action is generated
                                (this will be the case with prefixating morphology)
  --iterations=ITERATIONS       if align-smart, use this number of iterations in the aligner [default: 150]
  --substitution                use substitution of y_i (for any x_i) as an action instead of (insert of y_i + delete)
  --copy-as-substitution        treat copy as substitutions?
  --param-tying                 use same embeddings for characters and actions inserting them
  --tag-wraps=WRAPS             wrap lemma and word with word boundary tags?
                                  both (use opening and closing tags)/close (only closing tag)/None [default: both]
  --verbose                     visualize results of internal evaluation, display train and dev set alignments
  --mode=MODE                   various operation modes of the trainer:
                                    eval (run evaluation without training)/mle (MLE training)/rl (reinforcement
                                    learning training) [default: mle]
  --alpha=ALPHA                 if mode==mrt, MRT distribution-smoothing parameter alpha [default: 0.05]
  --beta=BETA                   if mode==mrt or mode==rl, MRT normalized edit distance scaling factor [default: 1]
  --no-baseline                 if mode==rl, RL does not use baseline correction of reward
  --beam-width=WIDTH            beam width for beam-search decoding [default: 8]
  --beam-widths=WIDTHS          a comma-separated sequence of beam widths for the final (dev/test) decoding with beam search
  --pretrain-epochs=EPOCHS      number of epochs to pretrain the model with MLE training [default: 0]
  --pretrain-until=ACC          MLE pretraining stops as soon as training accuracy score ACC is reached [default: 0]
  --sample-size=SAMPLE-SIZE     if mode==mrt or mode=rl, number of samples drawn from the model per training sample [default: 20]
  --scale-negative=S            if mode==rl, scale negative rewards by S [default: 0.1]
  --test-path=TEST-PATH         test set path
  --reload-path=RELOAD-PATH     reload a pretrained model at this path (possibly relative to RESULTS-PATH)
  --hall-path=HALL-PATH         path with hallucinated data
"""

import random

import dynet as dy
import numpy as np
from docopt import docopt

from args_processor import process_arguments
from trainer import TrainingSession, dev_external_eval, test_external_eval
from util import evaluate_features_predictions, write_generalized_measures

# import sys, codecs

# sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
# sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
# sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)

if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)

    print(docopt(__doc__))

    print('Processing arguments...')
    arguments = process_arguments(docopt(__doc__))
    paths, data_arguments, model_arguments, optim_arguments = arguments

    print('Loading data... Dataset: {}'.format(data_arguments['dataset']))
    train_data = data_arguments['dataset'].from_file(paths['train_path'], **data_arguments)
    phonology_converter = train_data.phonology_converter
    model_arguments['phonology_converter'] = phonology_converter
    VOCAB = train_data.vocab
    VOCAB.train_cutoff()  # knows that entities before come from train set
    batch_size = optim_arguments['decbatch-size']

    if paths['dev_path']:
        dev_data = data_arguments['dataset'].from_file(paths['dev_path'], vocab=VOCAB, **data_arguments)
        dev_batches = [dev_data.samples[i:i + batch_size] for i in range(0, len(dev_data), batch_size)]
    else:
        dev_data = None
        dev_batches = []

    if paths['test_path']:
        # no alignments, hence BaseDataSet
        test_data = data_arguments['dataset'].from_file(paths['test_path'], vocab=VOCAB, **data_arguments)
        # test_data = PCFPDataSet.from_file(paths['test_path'], vocab=VOCAB, **data_arguments)
    else:
        test_data = None

    model = None

    if not optim_arguments['eval']:
        print('Building model for training... Transducer: {}'.format(model_arguments['transducer']))
        model = dy.Model()
        transducer = model_arguments['transducer'](model, VOCAB, **model_arguments)

        training_session = TrainingSession(model, transducer, VOCAB, train_data, dev_data,
                                           optim_arguments['batch-size'],  # train batchsize
                                           optim_arguments['optimizer'], batch_size, dev_batches)

        if paths['reload_path']:
            training_session.reload(paths['reload_path'], paths['tmp_model_path'])

        # region handle pretraining
        if optim_arguments['pretrain-epochs'] or optim_arguments['pretrain-until']:
            pretrain_epochs = optim_arguments['pretrain-epochs']
            train_until_accuracy = optim_arguments['pretrain-until']
            if pretrain_epochs:
                print('Pretraining the model in a supervised manner for {} epochs.'.format(pretrain_epochs))
            else:
                print(('Pretraining the model in a supervised manner until'
                       ' train accuracy {}.'.format(train_until_accuracy)))
            training_session.run_MLE_training(epochs=pretrain_epochs,
                                              train_until_accuracy=train_until_accuracy,
                                              max_patience=optim_arguments['patience'],
                                              pick_best_accuracy=optim_arguments['pick-acc'],
                                              dropout=optim_arguments['pretrain-dropout'],
                                              l2=optim_arguments['l2'],
                                              log_file_path=paths['log_file_path'],
                                              tmp_model_path=paths['tmp_model_path'],
                                              check_condition=data_arguments['verbose'])
            print('Finished pretraining. Train loss: {}'.format(training_session.avg_loss))
            print('Reloading the best supervised model...')
            training_session.reload(paths['tmp_model_path'])
        else:
            print('No supervised pretraining.')
        # endregion handle pretraining

        if optim_arguments['mode'] == 'mle':
            training_session.run_MLE_training(epochs=optim_arguments['epochs'],
                                              max_patience=optim_arguments['patience'],
                                              pick_best_accuracy=optim_arguments['pick-acc'],
                                              dropout=optim_arguments['dropout'],
                                              l2=optim_arguments['l2'],
                                              log_file_path=paths['log_file_path'],
                                              tmp_model_path=paths['tmp_model_path'],
                                              check_condition=data_arguments['verbose'])

        elif optim_arguments['mode'] == 'rl':
            training_session.run_RL_training(
                epochs=optim_arguments['epochs'],
                max_patience=optim_arguments['patience'],
                pick_best_accuracy=optim_arguments['pick-acc'],
                dropout=optim_arguments['dropout'],
                l2=optim_arguments['l2'],
                sample_size=optim_arguments['sample-size'],
                beta=optim_arguments['beta'],
                scale_negative=optim_arguments['scale-negative'],
                baseline=optim_arguments['baseline'],
                log_file_path=paths['log_file_path'],
                tmp_model_path=paths['tmp_model_path'],
                check_condition=data_arguments['verbose'])

        elif optim_arguments['mode'] == 'mrt':
            training_session.run_MRT_training(
                epochs=optim_arguments['epochs'],
                max_patience=optim_arguments['patience'],
                pick_best_accuracy=optim_arguments['pick-acc'],
                dropout=optim_arguments['dropout'],
                l2=optim_arguments['l2'],
                sample_size=optim_arguments['sample-size'],
                alpha=optim_arguments['alpha'],
                beta=optim_arguments['beta'],
                log_file_path=paths['log_file_path'],
                tmp_model_path=paths['tmp_model_path'],
                check_condition=data_arguments['verbose'])
        else:
            raise NotImplementedError('Unknown training mode.')
    else:
        print('Skipped training by request. Evaluating best models.')

    model = dy.Model()
    transducer = model_arguments['transducer'](model, VOCAB, **model_arguments)
    print('Trying to load model from: {}'.format(paths['tmp_model_path']))
    model.populate(paths['tmp_model_path'])

    if dev_data:
        print('=========DEV EVALUATION:=========')
        dev_external_eval(dev_batches, transducer, VOCAB, *arguments)

    if test_data:
        print('=========TEST EVALUATION:=========')
        test_batches = [test_data.samples[i: i + batch_size] for i in range(0, len(test_data), batch_size)]
        test_accuracy = test_external_eval(test_batches, transducer, VOCAB, paths,
                                           optim_arguments['beam-widths'], data_arguments['sigm2017format'])
    else:
        test_accuracy = -1

    # TODO: re-add test evaluation for no phonology mode.
    if model_arguments['use_phonology'] and test_accuracy != -1:
        # Reevaluate at graphemes level: Read the test predictions file and evaluate the features predictions.
        # Then, write in f.stats all the 4 measures.
        test_predictions_file = paths['test_output']('greedy') + 'predictions'
        measures = evaluate_features_predictions(test_predictions_file, phonology_converter,
                                                 output_mode='phonemes' if data_arguments['self_attention'] else 'features')

        assert measures[1] == test_accuracy  # the return of test_external_eval equals to the (features-level) predictions' evaluation
        write_generalized_measures(paths['stats_file_path'], measures)
