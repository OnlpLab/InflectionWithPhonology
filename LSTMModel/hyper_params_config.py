from argparse import ArgumentParser

data_types = {'g': 'graphemes', 'p': 'phonemes', 'f': 'features'}

# region HPs
# Training hyperparameters
num_epochs = 50 # orignal=50. Note: must be >1 !!!
learning_rate = 3e-4
batch_size = 32 # original=32
LR_patience = 6
LR_factor = 0.82

# Model hyperparameters
encoder_embedding_size = 300 # original=300
decoder_embedding_size = 300 # original=300
hidden_size = 256 # original=256
num_layers = 1
enc_dropout = 0.1 # 0.0 is equivalent to Identity function
dec_dropout = 0.1 # 0.0 is equivalent to Identity function
# endregion HPs

parser = ArgumentParser(description="Parse arguments for linguistic configuration")
parser.add_argument('lang', type=str, choices=['bul', 'fin', 'hun', 'kat', 'lav', 'sqi', 'swc', 'tur'], help="Language to be processed", nargs='?', default='kat')
parser.add_argument('POS', type=str, choices=['V','N','ADJ'], help="Part of speech to be processed", nargs='?', default='V')
parser.add_argument('training_mode', type=str, choices=['form', 'lemma'], help="Can be either form-split or lemma-split", nargs='?', default='form')
parser.add_argument('inp_phon_type', type=str, choices=['g','p','f'], help="Phonological representation of the input", nargs='?', default='g')
parser.add_argument('out_phon_type', type=str, choices=['g','p','f'], help="Phonological representation of the output", nargs='?', default='g')
parser.add_argument('analogy_type', type=str, choices=['src1_cross1', 'None'], help='The analogies type to be applied', nargs='?', default='None')
parser.add_argument('SEED', type=int, help='Initial seed for all random operations', nargs='?', default=42)
parser.add_argument('device_idx', type=str, help='GPU index', nargs='?', default='0')
parser.add_argument('--ATTN', action='store_true', help="If True and inp_phon_type=='f', input features are combined in a Self-Attention layer to form a single vector.", default=False)
args = parser.parse_args()
lang, POS, SEED, device_idx = args.lang, args.POS, args.SEED, args.device_idx
analogy_type = args.analogy_type
training_mode, inp_phon_type, out_phon_type = args.training_mode, args.inp_phon_type, args.out_phon_type
inp_phon_type, out_phon_type = data_types[inp_phon_type], data_types[out_phon_type]
ANALOGY_MODE = analogy_type!='None'

PHON_UPGRADED = inp_phon_type=='features'
PHON_REEVALUATE = out_phon_type != 'graphemes' # evaluation at the graphemes-level is also required
PHON_USE_ATTENTION = args.ATTN and PHON_UPGRADED # apply self-attention to the features when extracting the average (olny if PHON_UPGRADED)
