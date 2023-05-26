import os

import datasets
from Models import hacm, hacm_sub, haem, haem_sub, hard, transducer
from DataRelatedClasses.DataSets.EditDataSet import EditDataSet
from aligners import cls_align, dumb_align, smart_align
from defaults import DATA_PATH, LANGUAGES_LIST, NULL_ARGS, RESULTS_PATH
from typing import Callable


def process_paths(arguments):
    
    def check_path(path, arg_name, is_data_path=True, create=True):
        if not os.path.exists(path):
            prefix = DATA_PATH if is_data_path else RESULTS_PATH
            orig_path = path
            path = os.path.join(prefix, path)
            if is_data_path:
                if not os.path.exists(path):
                    print('{} incorrect: {} and {}'.format(arg_name, orig_path, path))
                    raise ValueError
            else:
                if os.path.exists(path):
                    print('Warning! Output path exists: {}'.format(path))
                elif create:
                    os.makedirs(path)
                    print('Created output path: {}'.format(path))
        elif not is_data_path:
            print('Warning! Output path exists: {}'.format(path))
        return path

    train_path = check_path(arguments['TRAIN-PATH'], 'TRAIN_PATH')
    if arguments['--hall-path']:
        hall_path = check_path(arguments['--hall-path'], 'hall_path')
        train_path = [train_path, hall_path]
    dev_path = check_path(arguments['DEV-PATH'], 'DEV_PATH')
    # dev_path = None
    if arguments['--test-path']:
        test_path = check_path(arguments['--test-path'], 'test_path')
    else:
        # indicates no test set eval should be performed
        test_path = None
    
    try:
        # if this is sigmorphon format:
        lang, _, regime = os.path.basename(train_path).rsplit('-', 2)
    except Exception:
        lang, regime = 'unk', 'unk'

    results_file_path = check_path(arguments['RESULTS-PATH'].replace('\r', ''), 'RESULTS_PATH', is_data_path=False)
    # some filenames defined from `results_file_path`
    log_file_path   = os.path.join(results_file_path, 'f.log')
    tmp_model_path  = os.path.join(results_file_path, 'f.model')
    stats_file_path = os.path.join(results_file_path, 'f.stats')
    # dec: this is decoding -- greedy or beam
    dev_output: Callable[[str], str]  = lambda dec: os.path.join(results_file_path, 'f.{}.dev.'.format(dec))
    test_output: Callable[[str], str] = lambda dec: os.path.join(results_file_path, 'f.{}.test.'.format(dec))

    if arguments['--reload-path'] == 'self':
        # flag to reload from result directory
        reload_path = tmp_model_path
    # elif arguments['--reload-path']:
    #     # reload path is relative to `RESULTS_PATH`
    #     # it's some possibly differently named model
    #     reload_path = None
    #     reload_dir = check_path(arguments['--reload-path'],
    #         'RESULTS_PATH', is_data_path=False, create=False)
    #     for p in os.listdir(reload_dir):
    #         if p.endswith('model'):
    #            reload_path = os.path.join(reload_dir, p)
    #            break
    #     if not reload_path:
    #         print 'Failed to find the model at this path: {}'.format(reload_dir)
    #         print 'Will skip model reload.'
    else:
        reload_path = None
    # reload_path = None

    return dict(lang=lang, regime=regime,
                train_path=train_path, dev_path=dev_path, test_path=test_path,
                results_file_path=results_file_path,
                tmp_model_path=tmp_model_path, log_file_path=log_file_path,
                stats_file_path=stats_file_path,
                dev_output=dev_output, test_output=test_output,
                reload_path=reload_path)


def process_data_arguments(arguments):
    train_file = arguments['TRAIN-PATH']
    assert len([lang for lang in LANGUAGES_LIST if lang in train_file]) == 1
    language = [lang for lang in LANGUAGES_LIST if lang in train_file][0]

    # Find which of the list
    if arguments['--align-dumb']:
        aligner = dumb_align
    elif arguments['--align-cls']:
        aligner = cls_align
    else:
        aligner = smart_align
        
    if arguments['--transducer'] in ['hacm', 'hard'] and \
        not (arguments['--substitution'] or arguments['--copy-as-substitution']):
        dset = datasets.MinimalDataSet # Careful, might be buggy
    else:
        dset = EditDataSet

    return {
        'language'      : language,
        'train_samples' : int(arguments['--train-samples']),
        'use_phonology' : arguments['--use-phonology'],
        'self_attention': arguments['--self-attn'] if arguments['--use-phonology'] else False,
        'dataset'       : dset,
        'aligner'       : aligner,
        'sigm2017format': arguments['--sigm2017format'],
        'no_feat_format': arguments['--no-feat-format'],
        'try_reverse'   : arguments['--try-reverse'],
        'verbose'       : 2 if arguments['--verbose'] else False,
        'iterations'    : int(arguments['--iterations']),
        'substitution'  : arguments['--substitution'],
        'copy_as_substitution' : arguments['--copy-as-substitution'],
        'pos_emb'       : arguments['--pos-emb'],
        'avm_feat_format' : arguments['--avm-feat-format'],
        'param_tying'   : arguments['--param-tying'],
        'tag_wraps'     : arguments['--tag-wraps'] if arguments['--tag-wraps'] not in NULL_ARGS else None
    }

def process_model_arguments(arguments):

    arg_transducer = arguments['--transducer']
    if arguments['--substitution'] or arguments['--copy-as-substitution']:
        # need a transducer that handles substitution actions
        if arg_transducer == 'hacm':
            transd = hacm_sub.MinimalTransducer
        else:
            transd = haem_sub.EditTransducer
    elif arg_transducer == 'hacm':
        transd = hacm.MinimalTransducer
    elif arg_transducer == 'stmx-haem':  # transduce return softmax, not log softmax probabilities!
        transd = haem.EditTransducer
    elif arg_transducer == 'hard':
        transd = hard.Transducer
    else:
        transd = transducer.Transducer

    return {
        'transducer'      : transd,
        'char_dim'        : int(arguments['--input']),
        'action_dim'      : int(arguments['--action-input']),
        'feat_dim'        : int(arguments['--feat-input']),
        'enc_hidden_dim'  : int(arguments['--enc-hidden']),
        'enc_layers'      : int(arguments['--enc-layers']),
        'dec_hidden_dim'  : int(arguments['--dec-hidden']),
        'dec_layers'      : int(arguments['--dec-layers']),
        'vanilla_lstm'    : arguments['--vanilla-lstm'],
        'mlp_dim'         : int(arguments['--mlp']),
        'nonlin'          : arguments['--nonlin'],
        'pos_emb'         : arguments['--pos-emb'],
        'avm_feat_format' : arguments['--avm-feat-format'],
        'lucky_w'         : int(arguments.get('--lucky-w', 55)),
        'param_tying'     : arguments['--param-tying'],
        'use_phonology' : arguments['--use-phonology'],
        'self_attention'  : arguments['--self-attn'] if arguments['--use-phonology'] else False
    }
    

def process_optimization_arguments(arguments):
    
    # for sanity / dev set checks
    beam_width = int(arguments['--beam-width'])

    # for eval purposes only
    beam_widths = []
    if arguments['--beam-widths']:
        beam_widths = [int(w) for w in arguments['--beam-widths'].split(',')]
    elif beam_width > 1:
        beam_widths = [beam_width]
    else:
        beam_widths = []

    dropout = float(arguments['--dropout'])
    pretrain_dropout = float(arguments['--pretrain-dropout']) if arguments['--pretrain-dropout'] else dropout
        
    return {
        'mode'            : arguments['--mode'],
        'eval'            : arguments['--mode'] == 'eval',
        'dropout'         : dropout,
        'pretrain-dropout': pretrain_dropout, 
        'optimizer'       : arguments['--optimization'],
        'l2'              : float(arguments['--l2']),
        'alpha'           : float(arguments['--alpha']),
        'beta'            : float(arguments['--beta']),
        'baseline'        : not arguments['--no-baseline'],
        'epochs'          : int(arguments['--epochs']),
        'patience'        : int(arguments['--patience']),
        'pick-acc'        : not arguments['--pick-loss'],
        'pretrain-epochs' : int(arguments['--pretrain-epochs']),
        'pretrain-until'  : float(arguments['--pretrain-until']),
        'batch-size'      : int(arguments['--batch-size']),
        'decbatch-size'   : int(arguments['--decbatch-size']),
        'sample-size'     : int(arguments['--sample-size']),
        'scale-negative'  : float(arguments['--scale-negative']),
        'beam-width'      : beam_width,
        'beam-widths'     : beam_widths} 

def process_arguments(arguments, verbose=True):
    
    paths = process_paths(arguments)
    data_arguments = process_data_arguments(arguments)
    model_arguments = process_model_arguments(arguments)
    optimization_arguments = process_optimization_arguments(arguments)
    
    if verbose:
        print()
        print('LANGUAGE: {}, REGIME: {}'.format(paths['lang'], paths['regime']))

        print('Train path:   {}'.format(paths['train_path']))
        print('Dev path:     {}'.format(paths['dev_path']))
        print('Test path:    {}'.format(paths['test_path']))
        print('Results path: {}'.format(paths['results_file_path']))
        print()

        for name, args in (('DATA ARGS:', data_arguments),
                           ('MODEL ARGS:', model_arguments),
                           ('OPTIMIZATION ARGS:', optimization_arguments)):
            print(name)
            for k, v in args.items():
                print('{:20} = {}'.format(k, v))
            print()

    return paths, data_arguments, model_arguments, optimization_arguments
