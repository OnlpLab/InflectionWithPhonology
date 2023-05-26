from itertools import product
from os import makedirs
from os.path import join, isdir

"""
Note: run this script above the folder "runs_scripts"
"""

pairs = """bul	ADJ
bul	V
fin	ADJ
fin	N
fin	V
hun	V
kat	N
kat	V
lav	N
lav	V
sqi	V
swc	ADJ
swc	V
tur	ADJ
tur	V""".split('\n')
pairs = list(enumerate([tuple(pair.split('\t')) for pair in pairs], 1))

# LANG=$1, POS=$2, SPLIT=$3, SEED=$4, TRAINSAMPLES=$5, MODE=$6
seeds = [42, 21, 100]
train_samples = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
modes = ['gg', 'ff', 'fp']

global_index = 0
for seed in seeds:
    bins = dict(zip(range(1, 8), [[], [], [], [], [], [], []]))
    directory_path = join("runs_scripts", "LearningCurves", f"{seed}")
    if not isdir(directory_path): makedirs(directory_path)
    for (pair, train_number, mode) in product(pairs, train_samples, modes):
        group_idx, (lang, pos) = pair
        samples_index = train_number // 1000
        command = f"bash dummy.sh {lang} {pos} lemma {seed} {train_number} {mode}\n"

        bin_idx = (samples_index - group_idx + 1) % 7 + 7 * int((samples_index - group_idx + 1) % 7 == 0)
        bins[bin_idx].append(command)
        global_index += 1

    for bin_idx, bin_commands in bins.items():
        with open(join(directory_path, f"bin{bin_idx}-{seed}.sh"), "w") as f:
            f.write(f"# bin #{bin_idx}\n\n")
            for i, command in enumerate(bin_commands):
                f.write(command)
                if i % 3 == 2: f.write(f"\n")

print(global_index)