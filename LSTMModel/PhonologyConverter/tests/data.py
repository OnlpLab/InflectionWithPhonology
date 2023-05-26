# This file contains toy data used for the UTs.

exact_match_examples = {'kat': 'არ მჭირდ-ებოდყეტ', 'swc': "magnchdhe-ong jwng'a",
                        'sqi': 'rdhëije rrçlldgj-ijdhegnjzh',
                        'lav': 'abscā t-raķkdzhēļšanģa', 'bul': 'най-ясюногщжто'}

# For the problematic languages, the expected difference is also attached
non_exact_match_examples = { 'hun': 'hűdályiokró- l eéfdzgycsklynndzso nyoyaxy',
                             'tur': 'yığmalılksar mveğateğwypûrtâşsmış',
                             'fin': 'ixlmksnngvnk- èeé aatööböyynyissä'}
expected_edit_distances = {'hun': {'p': 2, 'f': 2}, 'tur': {'p': 5, 'f': 6}, 'fin': {'p': 4, 'f': 4}}

# A real bulgarian prediction that caused a bug
bulgarian_prediction = ('2', '10', '18', '$', '24', '29', 'NA', '$', '33', 'NA', 'NA', '$', '2', '10', '18', '$',
                        '24', '29', 'NA', '$', '33', 'NA', 'NA', '$', '26', '28', '30', '$', '2', '16', '19', '$',
                        '33', 'NA', 'NA', '$', '2', '10', '18', '$', '24', '29', 'NA', '$', '33', 'NA', 'NA', '$', '$',
                        '33', 'NA', 'NA', '$', '2', '11', '19', '$', '8', '12', '19', '$', '26', '28', '30', '$',
                        '22', '27', 'NA', '$', '2', '16', '19')

stringified_range_with_NAs = [str(n) for n in list(range(1, 21))] + ['NA'] * 5

all_unknown_valid_prediction = ',$,'.join(stringified_range_with_NAs).split(',') # [1, $, 2, $, ..., $, 20, $, NA, ...]

all_unknown_invalid_prediction = ',$,$,'.join(stringified_range_with_NAs).split(',') # [1, $, $, 2, $, $, ..., $, $, 20, $, $, NA, ...]

all_unknown_expected_result = ''.join(['#'] * 25) # assuming '#' is the character for unknown features bundle
