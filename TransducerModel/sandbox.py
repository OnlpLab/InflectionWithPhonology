from os import listdir
from os.path import join, isfile, isdir
from shutil import move

valid_path, invalid_path = 'ValidRuns', 'InvalidRuns'

def count_valid_runs(results_folder, is_run_valid, fix=False):
    valids_counter, invalids_counter, valid_runs, invalid_runs = 0, 0, [], []
    sub_folders = [folder for folder in listdir(results_folder) if folder.startswith("Outputs_")]

    for sub_folder in sub_folders:
        folder_to_validate = join(results_folder, sub_folder)
        if is_run_valid(folder_to_validate):
            valid_runs.append(sub_folder)
            valids_counter += 1
            if fix: move(folder_to_validate, join(results_folder, valid_path, sub_folder))
        else:
            invalid_runs.append(sub_folder)
            invalids_counter += 1
            if fix: move(folder_to_validate, join(results_folder, invalid_path, sub_folder))

    valid_runs.sort()
    invalid_runs.sort()

    # region prints
    print("Valid runs:")
    for sub_folder in valid_runs:
        print(f'\t{sub_folder}')
    print(f'\t{valids_counter} valid runs.')

    print('Invalid runs:')
    for sub_folder in invalid_runs:
        print(f'\t{sub_folder}')
    print(f'\t{invalids_counter} invalid runs.')
    # endregion prints

if __name__ == '__main__':
    results_folder = join('.', 'Results', 'None_42_f_p_attn')

    def is_run_valid(folder_path):
        log_file = join(folder_path, 'f.log')
        if not isfile(log_file):
            print(f"No f.log file in {log_file}!")
            return False

        logs_lines = open(log_file).readlines()
        if '0.0' in ' '.join(logs_lines[-1].strip().split()[2:4]): # accuracy on train or dev in last epoch is 0
            print(f'Warning: check out {folder_path}/f.log')

        stats_file = join(folder_path, 'f.stats')

        return len(logs_lines) == 51 and isfile(stats_file)

    count_valid_runs(results_folder, is_run_valid)