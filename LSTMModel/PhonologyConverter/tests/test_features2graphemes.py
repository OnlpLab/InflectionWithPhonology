import unittest
from parameterized import parameterized_class
from PhonologyConverter.languages_setup import LanguageSetup
from data import exact_match_examples, bulgarian_prediction
from data import all_unknown_valid_prediction, all_unknown_invalid_prediction, all_unknown_expected_result

# Note: in order to run the test suite, you might need to redirect the phonemes.json path in g2p_config.py
@parameterized_class(('language', 'word'), [(language, word) for language, word in exact_match_examples.items()])
class Features2GraphemesTestCase(unittest.TestCase):
    def test_prefix_padding(self):
        phonology_converter = LanguageSetup.create_phonology_converter(self.language)
        features = phonology_converter.word2phonemes(self.word, mode='features')

        padded_features = ['$'] * 50 + features
        recons_normalized_word = phonology_converter.phonemes2word(padded_features, 'features', normalize=True)
        self.assertEqual(self.word, recons_normalized_word)

    def test_suffix_padding(self):
        phonology_converter = LanguageSetup.create_phonology_converter(self.language)
        features = phonology_converter.word2phonemes(self.word, mode='features')

        padded_features = features + ['$'] * 50
        recons_normalized_word = phonology_converter.phonemes2word(padded_features, 'features', normalize=True)
        self.assertEqual(self.word, recons_normalized_word)

    def test_infix_dollars_padding(self):
        phonology_converter = LanguageSetup.create_phonology_converter(self.language)
        features = phonology_converter.word2phonemes(self.word, mode='features')

        for pad_size in range(1, 31):
            comma_separated_features = ','.join(features)
            padded_comma_separated_features = comma_separated_features.replace('$', ",".join(["$"] * pad_size))
            padded_features = padded_comma_separated_features.split(',')
            recons_normalized_word = phonology_converter.phonemes2word(padded_features, 'features', normalize=True)
            self.assertEqual(self.word, recons_normalized_word)

    def test_all_dollar_paddings_affixes(self):
        phonology_converter = LanguageSetup.create_phonology_converter(self.language)
        features = phonology_converter.word2phonemes(self.word, mode='features')

        for pad_size in range(1, 31):
            comma_separated_features = ','.join(features)
            padded_comma_separated_features = comma_separated_features.replace('$', ",".join(["$"] * pad_size))
            padded_features = padded_comma_separated_features.split(',')

            for prefix_pad in range(1, 31):
                for suffix_pad in range(1, 31):
                    padded_features_combined = ['$'] * prefix_pad + padded_features + ['$'] * suffix_pad

                    recons_normalized_word = phonology_converter.phonemes2word(padded_features_combined, 'features', normalize=True)
                    self.assertEqual(self.word, recons_normalized_word)

class SpecificExamplesTestCase(unittest.TestCase):
    def test_all_unknown_predictions(self):
        converter = LanguageSetup.create_phonology_converter('fin') # the language doesn't matter

        valid_unknowns_conversion = converter.phonemes2word(all_unknown_valid_prediction, 'features')
        self.assertEqual(valid_unknowns_conversion, all_unknown_expected_result)

        invalid_unknowns_conversion = converter.phonemes2word(all_unknown_invalid_prediction, 'features', normalize=True)
        self.assertEqual(invalid_unknowns_conversion, all_unknown_expected_result)


    def test_specific_bul_prediction(self):
        bulgarian_converter = LanguageSetup.create_phonology_converter('bul')

        with self.assertRaises(ValueError): # ValueError: "invalid literal for int() with base 10: '$'". Caused by 2 following '$'s.
            bulgarian_converter.phonemes2word(bulgarian_prediction, 'features', normalize=False)

        # But this one *will* complete successfully
        bulgarian_word = bulgarian_converter.phonemes2word(bulgarian_prediction, 'features', normalize=True)
        self.assertEqual(type(bulgarian_word), str)
        # Although it's not a real word, the conversion still exists
        self.assertEqual(bulgarian_word, 'т# т# у# т#  з#уа#')


if __name__ == '__main__':
    unittest.main()
