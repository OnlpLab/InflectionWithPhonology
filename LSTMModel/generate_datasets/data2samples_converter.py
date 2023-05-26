import os

class Data2SamplesConverter:
    def __init__(self):
        def default_reinflection(line:[str]) -> ([[str]], [str]):
            # [src_feat, src_form, trg_feat, trg_form] => ([src_feat.split(';'), list(src_form), trg_feat.split(';')] , list(trg_form))
            src_feat, src_form, trg_feat, trg_form = line
            return [src_feat.split(";"), list(src_form), trg_feat.split(";")], list(trg_form) # e.g. [['V','s3',...], ['ე','რ','ქ',...], ['V','s2',...]], ['გ','ე',...]
        self.default_reinflection = default_reinflection

    def reinflection2sample(self, line:[str], parsing_func=None) -> (str, str):
        # Convert a line (2-element list) to a reinflection sample. Can be either standard or a special function (Phonology/Analogies)
        if parsing_func is None: parsing_func = self.default_reinflection
        src, trg = parsing_func(line)
        src = ',+,'.join([','.join(e) for e in src]) # '+' is the separator for morphological-level characters (graphemes and morph-features)
        trg = ','.join(trg)
        return src, trg

    def sample2reinflection(self, sample:([str], [str]), parsing_func=None) -> ([str], str):
        # Takes src & trg as lists of characters, parses them somehow and returns
        if parsing_func is None: parsing_func = self.default_reinflection
        src, trg = parsing_func(*sample)
        return src, trg


    def reinflection2TSV(self, fn, suffix='', old_dir=os.path.join(".data","OriginalData"),
                         new_dir='.data', parsing_func=None, data=None) -> str:
        """
        Convert a file in some Reinflection format (e.g. src_feat\tsrc_form\ttrg_feat\ttrg_form) to
        a TSV file of the format src\ttrg, each one consists of CSV strings constructed by a given parsing function.
        """
        fn = os.path.join(old_dir,fn)
        new_fn = os.path.splitext(fn.replace(old_dir,new_dir))[0] + suffix + ".tsv"

        if data is None:
            data = open(fn, encoding='utf8').read().split('\n') # without using "with open.."
            data = [line.split('\t') for line in data]
            data = list(filter(lambda x: x != [""], data)) # remove empty strings if exist

        examples = []
        for e in data:
            src, trg = self.reinflection2sample(e, parsing_func=parsing_func)
            examples.append(f"{src}\t{trg}")

        open(new_fn, mode='w', encoding='utf8').write('\n'.join(examples))
        return new_fn
