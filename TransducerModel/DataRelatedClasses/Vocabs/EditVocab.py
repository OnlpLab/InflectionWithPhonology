from DataRelatedClasses.Vocabs.VocabBox import VocabBox
from defaults import (UNK, UNK_CHAR, BEGIN_WORD, BEGIN_WORD_CHAR,
    END_WORD, END_WORD_CHAR, DELETE, DELETE_CHAR, COPY, COPY_CHAR)

class EditVocab(VocabBox):
    def __init__(self, encoding=None):
        acts = {UNK_CHAR : UNK,
                BEGIN_WORD_CHAR : BEGIN_WORD,
                END_WORD_CHAR : END_WORD,
                DELETE_CHAR : DELETE,
                COPY_CHAR : COPY}
        super(EditVocab, self).__init__(acts, encoding)
