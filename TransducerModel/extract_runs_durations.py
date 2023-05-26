from pandas import DataFrame, read_excel
import os
from datetime import datetime
from os import remove
from os.path import join, basename, getmtime, isdir

def get_difference(folder_name, verbose = False):
    file_name = join(folder_name, "f.log")
    return get_folder_difference(file_name, verbose=verbose)

def get_folder_difference(folder_name, verbose = False):
    creation_datetime = datetime.strptime(basename(folder_name)[8:25], '%Y-%m-%d %H%M%S')
    if verbose: print(creation_datetime)
    modification_datetime = datetime.fromtimestamp(getmtime(folder_name))
    if verbose: print(modification_datetime)
    delta = modification_datetime - creation_datetime
    delta_str = str(delta).split('.')[0]
    if verbose: print(delta)
    return delta_str, delta.seconds

def extract_durations(results_folder, excel_file):
    if excel_file in os.listdir(): remove(excel_file)

    accuracies = []
    dirs = os.listdir(results_folder)
    dirs.sort()
    for sub_folder in dirs:
        lang, pos, split = basename(sub_folder).split("_")[2:5]
        delta, total_seconds = get_folder_difference(join(results_folder, sub_folder))
        accuracies.append([lang, pos, split, delta, total_seconds])

    df = DataFrame(accuracies, columns=['Language', 'POS', 'Split', 'Duration', 'In Seconds'])
    df = df.fillna("")
    df.to_excel(excel_file)

def main():
    df = read_excel("Durations.xlsx", "runs_list")
    seed = "100"
    runs_scripts_folder = join("runs_scripts", "f_f_runs", seed)
    if not isdir(runs_scripts_folder): os.makedirs(runs_scripts_folder)

    curr_index = -1
    for j in range(df.shape[0]):
        lines_to_write = []
        row = df.loc[j]
        Index = row.Index
        if curr_index != Index:
            curr_index = Index
            lines_to_write.append(f"# batch {Index}")

        lines_to_write.append(f"bash dummy.sh {row.Language} {row.POS} {row.Split} {seed}")

        for line in lines_to_write:
            with open(join(runs_scripts_folder, f"group{Index}-{seed}.sh"), "a+") as file:
                file.write(line+'\n')


if __name__ == '__main__':
    # main()
    results_folder = join("Results", "None_42_f_p_attn")
    excel_file = 'Durations_42_f_p_attn.xlsx'
    extract_durations(results_folder, excel_file)