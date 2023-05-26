__author__ = "David Guriel"
import json
import os

place = ['labial', 'dental', 'alveolar', 'velarized-alveolar', 'post-alveolar', 'velar', 'uvular', 'glottal', 'palatal'] # 0-8
manner = ['nasal', 'plosive', 'fricative', 'affricate', 'trill', 'tap', 'lateral', 'approximant', 'implosive'] # 9-17
voice = ['voiceless', 'voiced', 'ejective', 'aspirated'] # 18-21
# Vowels features:
height = ['open', 'open-mid', 'mid', 'close-mid', 'close'] # 22-26
backness = ['front', 'back', 'central'] # 27-29
roundness = ['rounded', 'unrounded'] # 30-31
length = ['long'] # only for vowels; no occurence means short vowel # 32
punctuations = [' ', '-', "'", "̇", '.', '*', '?'] # 33-39
# '*' is for predictions of non-existent feature bundles (see languages_setup.py). Note: this line differs from the original g2p_config file!

phon_features = place + manner + voice + height + backness + roundness + length + punctuations
idx2feature = dict(enumerate(phon_features))
feature2idx = {v:k for k, v in idx2feature.items()} # => {'labial':0,...,'nasal':6,...,'front':18,'back':19}

# for writing the dictionaries, use the command: json.dump({"vowels":p2f_vowels_dict, "consonants":p2f_consonants_dict}, open("phonemes.json","w",encoding='utf8'), indent=2)
phonemes = json.load(open(os.path.join(os.path.dirname(__file__), "phonemes.json"), encoding='utf8'))
p2f_consonants = {k:tuple(v) for k,v in phonemes['consonants'].items()}
p2f_vowels     = {k:tuple(v) for k,v in phonemes['vowels'].items()}
p2f_vowels.update({ k+'ː': v+('long',) for k,v in p2f_vowels.items()}) # account for long vowels (and double their number)
punctuations_g2p_dict = dict(zip(punctuations, punctuations))
p2f_punctuations_dict = dict(zip(punctuations, [tuple(f) for f in punctuations]))
p2f_dict = {**p2f_vowels, **p2f_consonants, **p2f_punctuations_dict}
f2p_dict = {v:k for k,v in p2f_dict.items()}
allowed_phoneme_tokens = tuple(p2f_dict.keys())
def feature_in_letter(feature:str, some_g2p_dict:[str], g:str): return feature in p2f_dict[some_g2p_dict[g]]

# region definedLangs
# Notes:
# - The structure of lang_components: [lang_g2p_dict, manual_word2phonemes, manual_phonemes2word, lang_clean_sample] - the last is a method for cleaning a line sample.
# - If you wish to add a new language, and a grapheme is mapped to more than a single phoneme (e.g. 'x' -> /ks/), make sure list(grapheme) results in real phonemes.
# region Georgian - kat
kat_alphabet = ['ა', 'ბ', 'გ', 'დ', 'ე', 'ვ', 'ზ', 'თ', 'ი', 'კ', 'ლ', 'მ', 'ნ', 'ო', 'პ', 'ჟ', 'რ', 'ს', 'ტ', 'უ', 'ფ', 'ქ', 'ღ', 'ყ', 'შ', 'ჩ', 'ც', 'ძ', 'წ', 'ჭ', 'ხ', 'ჯ', 'ჰ']
kat_phonemes = ['ɑ', 'b', 'ɡ', 'd', 'ɛ', 'v', 'z', 'tʰ', 'i', 'kʼ', 'l', 'm', 'n', 'ɔ', 'pʼ', 'ʒ', 'r', 's', 'tʼ', 'u', 'pʰ', 'kʰ', 'ɣ', 'qʼ', 'ʃ', 't͡ʃʰ', 't͡sʰ', 'd͡z', 't͡sʼ', 't͡ʃʼ', 'x', 'd͡ʒ', 'h']
kat_g2p_dict = dict(zip(kat_alphabet, kat_phonemes))
kat_components = [kat_g2p_dict, None, None, None]
# endregion Georgian - kat


# region Swahili - swc
swc_alphabet = ['a', 'e', 'i', 'o', 'u',  'b', 'ch', 'd', 'dh', 'f', 'g', 'gh', 'h', 'j', 'k', 'kh', 'l', 'm', 'n', 'ng', "ng'", 'ny', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z']
swc_phonemes = ['ɑ', 'ɛ', 'i', 'ɔ', 'u',  'ɓ', 't͡ʃ', 'ɗ', 'ð', 'f', 'ɠ', 'ɣ', 'h', 'ʄ', 'k', 'x', 'l', 'm', 'n', 'ɡ', 'ŋ', 'ɲ', 'p', 'r', 's', 'ʃ', 't', 'θ', 'v', 'w', 'j', 'z']
swc_g2p_dict = dict(zip(swc_alphabet, swc_phonemes))
def swc_word2phonemes(w:[str]):
    return word2phonemes_with_trigraphs(w, lang='swc')
swc_components = [swc_g2p_dict, swc_word2phonemes, None, None]
# endregion Swahili - swc


# region Albanian - sqi
sqi_alphabet = ['a', 'b', 'c', 'ç', 'd', 'dh', 'e', 'ë', 'f', 'g', 'gj', 'h', 'i', 'j', 'k', 'l', 'll', 'm', 'n', 'nj', 'o', 'p', 'q', 'r', 'rr', 's', 'sh', 't', 'th', 'u', 'v', 'x', 'xh', 'y', 'z', 'zh']
sqi_phonemes = ['a', 'b', 't͡s', 't͡ʃ', 'd', 'ð', 'ɛ', 'ə', 'f', 'ɡ', 'ɟ͡ʝ', 'h', 'i', 'j', 'k', 'l', 'ɫ', 'm', 'n', 'ɲ', 'ɔ', 'p', 'c', 'ɹ', 'r', 's', 'ʃ', 't', 'θ', 'u', 'v', 'd͡z', 'd͡ʒ', 'y', 'z', 'ʒ']
sqi_g2p_dict = dict(zip(sqi_alphabet, sqi_phonemes))
def sqi_word2phonemes(w:[str]):
    return word2phonemes_with_digraphs(w, lang='sqi')
def sqi_clean_sample(x: str) -> str:
    return x.replace("',","") # appears in the data only as part of "për t'u ..." (NFIN)
sqi_components = [sqi_g2p_dict, sqi_word2phonemes, None, sqi_clean_sample]
# endregion Albanian - sqi


# region Latvian - lav
lav_alphabet = ['a', 'ā', 'e',  'ē', 'i',  'ī', 'o', 'u',  'ū', 'b',  'c',  'č', 'd', 'dz', 'dž', 'f', 'g', 'ģ', 'h', 'j', 'k', 'ķ', 'l', 'ļ', 'm', 'n', 'ņ', 'p', 'r', 's', 'š', 't', 'v', 'z', 'ž']
lav_phonemes = ['ɑ', 'ɑː', 'e', 'eː', 'i', 'iː', 'o', 'u', 'uː', 'b', 't̪͡s̪', 't͡ʃ', 'd̪', 'd̪͡z̪', 'd͡ʒ', 'f', 'ɡ', 'ɟ', 'x', 'j', 'k', 'c', 'l', 'ʎ', 'm', 'ŋ', 'ɲ', 'p', 'r', 's', 'ʃ', 't̪', 'v', 'z', 'ʒ']
lav_g2p_dict = {**dict(zip(lav_alphabet, lav_phonemes)), **punctuations_g2p_dict}
lav_p2g_dict = {**dict(zip(lav_phonemes, lav_alphabet)), **punctuations_g2p_dict}
def lav_phonemes2word(phonemes:[str]):
    return phonemes2graphemes_with_doubles(phonemes, lang='lav')
def lav_clean_sample(x:str) -> str:
    return x.replace('y','i').replace('í', 'ī').replace('ŗ', 'r').replace("LgSPEC8", "LGSPEC8") # replace the 3 occurences of 'í' with 'ī', and the 28 occ. of 'ŗ'
lav_components = [lav_g2p_dict, None, lav_phonemes2word, lav_clean_sample]
# endregion Latvian - lav


# region Bulgarian - bul
bul_alphabet = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь', 'ю', 'я']
bul_phonemes = ['a', 'b', 'v', 'ɡ', 'd', 'ɛ', 'ʒ', 'z', 'i', 'j', 'k', 'l', 'm', 'n', 'ɔ', 'p', 'r', 's', 't', 'u', 'f', 'x', 't͡s', 't͡ʃ', 'ʃ', 'ʃt', 'ɤ', 'j', 'ju', 'ja']
bul_g2p_dict = {**dict(zip(bul_alphabet, bul_phonemes)), **punctuations_g2p_dict}
bul_p2g_dict = {**dict(zip(bul_phonemes, bul_alphabet)), **punctuations_g2p_dict}
def bul_word2phonemes(w:[str]):
    phonemes = []
    for g in w:
        target_phoneme = [*bul_g2p_dict[g]] if g in {'щ', 'ю', 'я'} else bul_g2p_dict[g]
        phonemes.extend(target_phoneme if type(target_phoneme)==list else [target_phoneme])
    return phonemes
def bul_phonemes2word(phonemes:[str]):
    special_mappings = {'j': 'й'} # just never map /j/ to 'ь' ('ь' - 151 vs 'й' - 7217)
    return phonemes2graphemes_with_doubles(phonemes, lang='bul', special_mappings=special_mappings)
bul_components = [bul_g2p_dict, bul_word2phonemes, bul_phonemes2word, None]
# endregion Bulgarian - bul


# region Hungarian - hun
hun_alphabet = ['a', 'á', 'b', 'c', 'cs', 'd', 'dz', 'dzs', 'e', 'é', 'f', 'g', 'gy', 'h', 'i', 'í', 'j', 'k', 'l', 'ly', 'm', 'n', 'ny', 'o', 'ó', 'ö', 'ő', 'p', 'r', 's', 'sz', 't', 'ty', 'u', 'ú', 'ü', 'ű', 'v', 'w', 'x', 'y', 'z', 'zs']
hun_phonemes = ['ɒ', 'aː', 'b', 't͡s', 't͡ʃ', 'd', 'd͡z', 'd͡ʒ', 'ɛ', 'eː', 'f', 'ɡ', 'ɟ', 'h', 'i', 'iː', 'j', 'k', 'l', 'ʎ', 'm', 'n', 'ɲ', 'o', 'oː', 'ø', 'øː', 'p', 'r', 'ʃ', 's', 't', 'c', 'u', 'uː', 'y', 'yː', 'v', 'w', 'ks', 'i', 'z', 'ʒ']
hun_g2p_dict = {**dict(zip(hun_alphabet, hun_phonemes)), **punctuations_g2p_dict}
hun_p2g_dict = {**dict(zip(hun_phonemes, hun_alphabet)), **punctuations_g2p_dict}
def hun_word2phonemes(w:[str]):
    return word2phonemes_with_trigraphs(w, 'hun')
def hun_phonemes2word(phonemes:[str]):
    special_mappings = {'i': 'i'} # just never map /i/ to 'y' ('y' - 173k vs 'i' - 628k)
    return phonemes2graphemes_with_doubles(phonemes, lang='hun', special_mappings=special_mappings)
def hun_clean_sample(x:str) -> str:
    # the " |or| " is a bug of the scraping from Wiktionary. It can appear at the end of a form
    # (search in the data for "jósolj|or|") or between 2 forms (search for "jóslok |or| jósolok").
    # There are also pipes ("|"), alone or preceded by a space " |".
    # The input is in format of ','.join(input), so the cleaning patterns follow this method.
    chars_to_remove = [','+','.join(" |or| "), ','+','.join("|or|"), ", ,|", ",|"]
    for p in chars_to_remove: x = x.replace(p, "")
    return x
hun_components = [hun_g2p_dict, hun_word2phonemes, hun_phonemes2word, hun_clean_sample]
# endregion Hungarian - hun


# region Turkish - tur
tur_alphabet = ['a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z', 'w', 'x' , 'â', 'î', 'û']
tur_phonemes = ['a', 'b', 'd͡ʒ', 't͡ʃ', 'd', 'ɛ', 'f', 'ɡ', 'j', 'h', 'ɯ', 'i', 'ʒ', 'k', 'l', 'm', 'n', 'o', 'œ', 'p', 'ɾ', 's', 'ʃ', 't', 'u', 'y', 'v', 'j', 'z', 'w', 'ks', 'aː', 'iː', 'uː']
tur_g2p_dict = {**dict(zip(tur_alphabet, tur_phonemes)), **punctuations_g2p_dict}
tur_p2g_dict = {**dict(zip(tur_phonemes, tur_alphabet)), **punctuations_g2p_dict}
def lengthen_last_item(phonemes_list):
    phonemes_list[-1]+='ː'
    return phonemes_list

turkish_vowels = {'a', 'e', 'i', 'o', 'u', 'ı', 'ö', 'ü', 'â', 'î', 'û'}
turkish_vowels_phonemes = set([tur_g2p_dict[g] for g in turkish_vowels])
def is_tur_vowel(c): return c in turkish_vowels
def is_tur_vowel_phoneme(c): return c in turkish_vowels_phonemes
# [aeiouıöüâîû]ğ
def tur_word2phonemes(graphemes:[str]):
    # Turkish has no digraphs, but the conversion of 'ğ' is a little complex, so we don't use word2phonemes_with_digraphs
    phonemes, i = [], 0
    while i < len(graphemes):
        g, resulted_phoneme = graphemes[i], ['']
        if g == 'ğ': # the previous character must be a vowel -- they must obey the regex [aeiouıöüâîû]ğ
            if i==len(graphemes)-1 or graphemes[i+1]==' ': # last letter before whitespace
                phonemes = lengthen_last_item(phonemes)
            else:
                trigraph = graphemes[i - 1: i + 2]
                if is_tur_vowel(trigraph[0]) and is_tur_vowel(trigraph[2]):
                    if trigraph[0] == trigraph[2]:
                        phonemes = lengthen_last_item(phonemes)
                        i += 1
                    # resulted_phoneme = [''] # unnecessary
                elif trigraph[0] == 'e': # i.e. trigraph[2] is not a vowel
                    resulted_phoneme = ['j']
                else:
                    phonemes = lengthen_last_item(phonemes)
        elif g == 'x':
            resulted_phoneme = ['k', 's'] # there are only 48 'x' occurences, so will be inversed to ['k', 's']!
        else:
            resulted_phoneme = [tur_g2p_dict[g]]
        if resulted_phoneme != ['']:
            phonemes.extend(resulted_phoneme)
        i += 1
    return phonemes

def tur_phonemes2word(phonemes:[str]):
    special_mappings = {'j': 'y', 'aː': 'â', 'iː': ['i', 'ğ', 'i'], 'uː': ['u', 'ğ', 'u'], 'ɛː': ['e', 'ğ']}
    graphemes, i = [], 0
    while i < len(phonemes):
        p = phonemes[i]
        if p in special_mappings:
            g = special_mappings[p]
        elif 'ː' in p: # a long vowel
            g = [tur_p2g_dict[p[0]], 'ğ']
            if i < len(phonemes)-1 and phonemes[i+1] != ' ':
                g += tur_p2g_dict[p[0]] # add the other grapheme only if it's not the last phoneme.
        elif is_tur_vowel_phoneme(p) and i < len(phonemes)-1 and is_tur_vowel_phoneme(phonemes[i+1]): # p and its follower are distinct vowels
            g = [tur_p2g_dict[p], 'ğ', tur_p2g_dict[phonemes[i+1]]]
            i += 1
        else:
            g = tur_p2g_dict[p]
        graphemes.extend(g if type(g)==list else [g])
        i += 1
    return graphemes
def tur_clean_sample(x:str) -> str:
    return x.replace('İ', 'i').replace('i̇', 'i')
tur_components = [tur_g2p_dict, tur_word2phonemes, tur_phonemes2word, tur_clean_sample]
# endregion Turkish - tur


# region Finnish - fin
fin_alphabet = ['a', 'è', 'é', 'e', 'i', 'o', 'u', 'y', 'ä', 'ö', 'b', 'c', 'd', 'f', 'g', 'ng', 'nk', 'h', 'j', 'k', 'l', 'm', 'n',
                'p', 'q', 'r', 's', 'š', 't', 'v', 'w', 'x', 'z', 'ž', 'å', 'aa', 'ee', 'ii', 'oo', 'uu', 'yy', 'ää', 'öö']
fin_phonemes = ['ɑ', 'e', 'e', 'e', 'i', 'o', 'u', 'y', 'a', 'ø', 'b', 's', 'd', 'f', 'ɡ', 'ŋŋ', 'ŋ', 'h', 'j', 'k', 'l', 'm', 'n',
                'p', 'k', 'r', 's', 'ʃ', 't', 'v', 'v', 'ks', 't͡s', 'ʒ', 'oː', 'ɑː', 'eː', 'iː', 'oː', 'uː', 'yː', 'aː', 'øː']
fin_g2p_dict = {**dict(zip(fin_alphabet, fin_phonemes)), **punctuations_g2p_dict}
fin_p2g_dict = {**dict(zip(fin_phonemes, fin_alphabet)), **punctuations_g2p_dict}
def fin_word2phonemes(w:[str]):
    return word2phonemes_with_digraphs(w, 'fin')
def fin_phonemes2word(phonemes:[str]):
    special_mappings = {
        'e': 'e', # ignore 'é' and 'è'
        'oː': ['o', 'o'], # ignore 'å'
        'k': 'k', # ignore 'q'
        's': 's', # ignore 'c'
        'v': 'v',} # ignore 'w'
    result = phonemes2graphemes_with_doubles(phonemes, lang='fin', special_mappings=special_mappings)
    result = list(''.join(result).replace('x','ks'))
    return result
def fin_clean_sample(x:str) -> str:
    chars_to_remove = ['\xa0', ":", "/", ","]
    for p in chars_to_remove: x = x.replace(p, "")
    x = x.replace("á", "a").replace("â", "a").replace("û", "u").replace("ü", "u")
    return x
fin_components = [fin_g2p_dict, fin_word2phonemes, fin_phonemes2word, fin_clean_sample]
# endregion Finnish - fin
# endregion definedLangs

langs_properties = {'bul': bul_components, 'fin': fin_components, 'hun': hun_components, 'kat': kat_components,
                    'lav': lav_components, 'sqi': sqi_components, 'swc': swc_components, 'tur': tur_components}
for k in langs_properties:
    if langs_properties[k][3] is None:
        langs_properties[k][3] = lambda x: x

def word2phonemes_with_digraphs(w:[str], lang:str, allowed_phoneme_tokens:[str] = allowed_phoneme_tokens) -> [str]:
    # Convert the graphemes to a list of phonemes according to the langauge's digraphs
    g2p_mapping = langs_properties[lang][0]
    digraphs = list(filter(lambda g: len(g) == 2, g2p_mapping.keys()))
    phonemes, i, flag = [], 0, False
    while i < len(w):
        if i < len(w)-1 and w[i]+w[i+1] in digraphs:
            phoneme_token = g2p_mapping[w[i] + w[i + 1]]
            i += 1
        else:
            phoneme_token = g2p_mapping[w[i]]
        if phoneme_token not in allowed_phoneme_tokens: # need to decompose to real phonemes
            phoneme_token = list(phoneme_token)
        phonemes.extend(phoneme_token if type(phoneme_token) == list else [phoneme_token])
        i += 1
    return phonemes

def word2phonemes_with_trigraphs(w:[str], lang:str, allowed_phoneme_tokens:[str] = allowed_phoneme_tokens) -> [str]:
    # Convert the graphemes to a list of phonemes according to the langauge's trigraphs & digraphs
    g2p_mapping = langs_properties[lang][0]
    digraphs = list(filter(lambda x: len(x)==2, g2p_mapping.keys()))
    trigraphs = list(filter(lambda x: len(x)==3, g2p_mapping.keys()))
    phonemes, i, flag = [], 0, False
    while i < len(w):
        if i < len(w)-2 and w[i]+w[i+1]+w[i+2] in trigraphs:
            phoneme_token = g2p_mapping[w[i]+w[i+1]+w[i+2]]
            i += 2
        elif i < len(w)-1 and w[i]+w[i+1] in digraphs:
            phoneme_token = g2p_mapping[w[i] + w[i + 1]]
            i += 1
        else:
            phoneme_token = g2p_mapping[w[i]]
        if phoneme_token not in allowed_phoneme_tokens: # need to decompose to real phonemes
            phoneme_token = list(phoneme_token)
        phonemes.extend(phoneme_token if type(phoneme_token) == list else [phoneme_token])
        i += 1
    return phonemes

def phonemes2graphemes_with_doubles(w:[str], lang:str, special_mappings:dict = None) -> [str]:
    # Convert the phonemes to a list of graphemes according to the langauge's double-phonemes.
    # special_mappings is a dictionary that intentionally ignores other possible graphemes at the g2p dictionary.
    p2g_mapping = {v:k for k,v in langs_properties[lang][0].items()}
    phoneme_doubles = list(filter(lambda x: len(x) > 1 and x not in allowed_phoneme_tokens, p2g_mapping.keys()))
    graphemes, i, flag = [], 0, False
    while i < len(w):
        if i < len(w)-1 and w[i]+w[i+1] in phoneme_doubles:
            grapheme_token = p2g_mapping[w[i] + w[i + 1]]
            i += 1
        else:
            if special_mappings is not None and w[i] in special_mappings: # assuming the sepcial mappings occur only in single real phonemes.
                grapheme_token = special_mappings[w[i]]
            else:
                grapheme_token = p2g_mapping[w[i]]
        grapheme_token = list(grapheme_token)
        graphemes.extend(grapheme_token)
        i += 1
    return graphemes

# for further debugging purposes:
def is_g2p_1to1(d:dict): return len(d.values())==len(set(d.values()))
def are_there_phonemes_unincluded_intheJSON(lang_phonemes:[str]) -> [str]: return [p for p in lang_phonemes if p not in p2f_dict.keys()]

# region manually inserting g-p pairs
# (copy that section to the console and insert; once done, copy back to the script.
# k,v, d = 'a', 'ɑ', {} # initial pair
# while (k,v)!=('',''):
#     d[k]=v
#     k,v = input("insert k"), input("insert v")
# endregion manually inserting g-p pairs:
# region analyze characters in the data
# def analyze_characters_in_data(lang):
#     from os.path import join
#     vocab = list(set(list(open(join(".data", "RawData", f"{lang}.txt"), encoding='utf8').read())))
#     vocab.sort()
#     print(f'Characters in {lang} data:')
#     print(vocab, '\n')
# for l in ['kat', 'swc', 'sqi', 'lav', 'bul', 'hun', 'tur', 'fin']:
#     analyze_characters_in_data(l)
# endregion analyze characters in the data
