import os
from os.path import join, isdir, isfile, getsize
from itertools import product

def are_substrings_in_string(target_string: str, substrings: tuple) -> bool:
    return all([substring in target_string for substring in substrings])

test_mode = True
logs_folder = join(".", "Results", "Logs", "Tests" if test_mode else "")

analogy_types, seeds = ['None', 'src1_cross1'], [42, 21, 7]
groups_indices = list(range(1,8))

lang_pos_groups = dict(zip(groups_indices, [['kat_V', 'kat_N', 'fin_ADJ'], ['swc_V', 'swc_ADJ', 'fin_V'], ['sqi_V', 'hun_V'], ['bul_V', 'bul_ADJ'], ['lav_V', 'lav_N'], ['tur_V', 'tur_ADJ'], ['fin_N']]))
    
def assert_folders_exist():
    for analogy, seed in product(analogy_types, seeds):
        assert isdir(f"{analogy}_{seed}"), f"Folder {analogy}_{seed} doesn't exist!"
        for group_idx in groups_indices:
            group_folder = join(f"{analogy}_{seed}", f"Group {group_idx}")
            assert isdir(group_folder), f"Folder {group_folder} doesn't exist!"

def count_files_in_folders(show_warnings):
    for analogy, seed in product(analogy_types, seeds):
        analogy_seed_folder = f"{analogy}_{seed}"
        for group_idx in groups_indices:
            expected_no_files = len(lang_pos_groups[group_idx]) * 6

            group_folder = join(analogy_seed_folder, f"Group {group_idx}")
            actual_files = [join(group_folder, f) for f in os.listdir(group_folder) if isfile(join(group_folder, f))]

            if show_warnings:
                for f in actual_files:
                    if getsize(f) < 30000:
                        print(f"Warning: potential problem in {f} in {group_folder}!")

            actual_no_files = len(actual_files) # excludes sub-folders like 'Bad phonemes.json results'
            if actual_no_files < expected_no_files:
                print(f"Found {actual_no_files} files in {group_folder} (less than {expected_no_files})!")
            elif actual_no_files > expected_no_files:
                print(f"Found {actual_no_files} files in {group_folder} (more than {expected_no_files})!")


def extract_results_from_file(file_path):
    # return the format morph_ED, morph_Acc, phon_ED, phon_Acc. If it's output mode is g,
    # then the last 2 should be None (recall the line PHON_REEVALUATE = out_phon_type != 'graphemes')
    lines = open(file_path, encoding='utf8').readlines()
    assert "Morphological level:" in lines[-3], "Results don't exist!"
    def find_2_floats_in_line(line):
        import re
        a, b = re.findall("\d+\.\d+", line)
        return float(a), float(b)
    morph_ED, morph_Acc = find_2_floats_in_line(lines[-3]) # morph. results line
    if "Phonological level:" in lines[-4]: # phon. measures required
        phon_ED, phon_Acc = find_2_floats_in_line(lines[-4]) # phon. results line
    else:
        phon_ED, phon_Acc = None, None
    return morph_ED, morph_Acc, phon_ED, phon_Acc

def extract_from_valid_files(show_warnings: bool) -> list:
    results = []
    modes_to_check = [('form_g_g',), ('form_f_f',), ('form_f_p', 'attn'), ('lemma_g_g',), ('lemma_f_f',), ('lemma_f_p', 'attn')]

    for analogy, seed in product(analogy_types, seeds): # 6
        analogy_seed_folder = join('.', f"{analogy}_{seed}")

        for group_idx, lang_pos_list in lang_pos_groups.items(): # 7
            group_folder = join(analogy_seed_folder, f"Group {group_idx}")
            files_in_group_folder = [join(group_folder, f) for f in os.listdir(group_folder) if isfile(join(group_folder, f))] # excludes sub-folders like 'Bad phonemes.json results'

            for lang_pos in lang_pos_list: # 1-3
                lang, pos = lang_pos.split('_')

                lang_pos_files = [f for f in files_in_group_folder if lang_pos in f] # e.g. files with 'swc_ADJ', 6 in total

                found_modes = dict(zip(modes_to_check, [False]*len(modes_to_check)))
                if len(lang_pos_files) != 6:
                    print(f"There are {len(lang_pos_files)} files of {lang_pos} in {group_folder}")
                else:
                    for file in lang_pos_files: # 6
                        search_modes_match = [are_substrings_in_string(file, mode + (f"{analogy}_{seed}", lang, pos, "Logs")) for mode in modes_to_check] # returns a "True-hot" list
                        mode_matched = modes_to_check[search_modes_match.index(True)]
                        found_modes[mode_matched] = True
                        if show_warnings and getsize(file) < 30000:
                            print(f"Warning: potential problem in {file} in {lang_pos} in {group_folder}!")
                        else:
                            # print(file)
                            file_measures = extract_results_from_file(file)

                            training_mode, io_type = mode_matched[0].split('_', maxsplit=1) # e.g. 'form', 'g_g'
                            results.append([analogy, seed, lang, pos, training_mode, io_type, *file_measures])
                    if not all(found_modes):
                        print(f"Not all 6 modes exist: {lang_pos} in {group_folder}!")

    return results

if __name__ == '__main__':
    assert isdir(logs_folder), f"Folder Results doesn't exist!"
    os.chdir(logs_folder)
    show_warnings = False

    # 0. Make sure all the required folders exist
    assert_folders_exist()

    # 1. Count the #files in each group to assert len(group) == X, where X = length(lang_pos_groups[group_idx]) * 6.
    # Exclude 'Bad phonemes.json results' from the counting! Print any problems.
    count_files_in_folders(show_warnings)

    is_extraction_mode = True
    if is_extraction_mode:
        # 2. Assert all files in every folder the X files match the schema '{lang}_{pos}_.+_{analogy}_{seed}'. Print any problems.
        # 3. In the valid folders, extract the 2 or 4 measures (morph_ED, morph_Acc, phon_ED, phon_Acc).
        # 4. Save them in a list: [[analogy, seed, lang, pos, form, g_g, morph_ED, morph_Acc, phon_ED, phon_Acc]]
        #                                       6         15          6
        results_records = extract_from_valid_files(show_warnings)
        from pandas import DataFrame
        df = DataFrame(results_records, columns=['AnalogyMode', 'Seed', 'Language', 'POS', 'Form/Lemma', 'IO mode',
                                                 'morph_ED', 'morph_Acc', 'phon_ED', 'phon_Acc'])
        # 5. Ignore the None values of the g-g phonological measures
        df = df.fillna("")

        excel_results_file = "Test-Results None-src1_cross1 42-21-7.xlsx"
        # 6. Finally, save everything in an Excel file called "Results None-src1_cross1 42-21-7.xlsx"
        df.to_excel(excel_results_file)
    print("Finished. Bye")