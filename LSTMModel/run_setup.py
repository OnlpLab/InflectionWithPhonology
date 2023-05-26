# This file links between hyper_params_config.py and utils.py & main.py
from os import mkdir
from os.path import join, isdir
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

import hyper_params_config as hp

user_params_config = f"{hp.lang}_{hp.POS}_{hp.training_mode}_{hp.inp_phon_type[0]}_{hp.out_phon_type[0]}_{hp.analogy_type}_{hp.SEED}" \
                     f"_{hp.device_idx}{'_attn' if hp.PHON_USE_ATTENTION else ''}" # used below for files naming

user_params_config_to_print = f"""Run arguments configuration:
- language = {hp.lang}, part-of-speech = {hp.POS}
- split-type = {hp.training_mode}
- input_format = {hp.inp_phon_type}, output_format = {hp.out_phon_type}, phon_upgraded = {hp.PHON_UPGRADED}, phon_self_attention = {hp.PHON_USE_ATTENTION}
- analogy_mode = {hp.ANALOGY_MODE}{f", analogy_type = {hp.analogy_type}" if hp.ANALOGY_MODE else ''}"""

if hp.analogy_type == 'None':
    train_file = join(".data", "Reinflection", f"{hp.lang}.{hp.POS}", f"{hp.lang}.{hp.POS}.{hp.training_mode}.train.tsv")
    dev_file =   join(".data", "Reinflection", f"{hp.lang}.{hp.POS}", f"{hp.lang}.{hp.POS}.{hp.training_mode}.dev.tsv")
    test_file =  join(".data", "Reinflection", f"{hp.lang}.{hp.POS}", f"{hp.lang}.{hp.POS}.{hp.training_mode}.test.tsv") # used only in test_single_model.py
else: # hp.analogy_type == 'src1_cross1'
    train_file = join(".data", "Reinflection", f"{hp.lang}.{hp.POS}", "src1_cross1", f"{hp.lang}.{hp.POS}.{hp.training_mode}.train.src1_cross1.tsv")
    dev_file =   join(".data", "Reinflection", f"{hp.lang}.{hp.POS}", "src1_cross1", f"{hp.lang}.{hp.POS}.{hp.training_mode}.dev.src1_cross1.tsv")
    test_file =  join(".data", "Reinflection", f"{hp.lang}.{hp.POS}", "src1_cross1", f"{hp.lang}.{hp.POS}.{hp.training_mode}.test.src1_cross1.tsv") # used only in test_single_model.py


# region output folders
resultsFolder = "Results"
def safely_create_results_subfolders(names):
    full_paths = []
    for name in names:
        full_path = join(resultsFolder, name)
        if not isdir(full_path): mkdir(full_path)
        full_paths.append(full_path)
    return full_paths
evaluation_graphs_folder, prediction_files_folder, model_checkpoints_folder, logs_folder, summaryWriter_logs_folder = \
    safely_create_results_subfolders(["EvaluationGraphs", "PredictionFiles", "Checkpoints", "Logs", "SummaryWriterLogs"])
# endregion output folders

def get_time_now_str(allow_colon:bool):
    from datetime import datetime
    s = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return s if allow_colon else s.replace(':', '')
time_stamp = get_time_now_str(allow_colon=False) # the time when the run started
user_params_with_time_stamp = f"{time_stamp}_{user_params_config}" # the ID of this run (more specific than time_stamp),
# unique unless if 2 identical runs are started at the same time

# region output files
evaluation_graphs_file = join(evaluation_graphs_folder, f"EvaluationGraph_{user_params_with_time_stamp}.png")
model_checkpoint_file =  join(model_checkpoints_folder, f"Model_Checkpoint_{user_params_with_time_stamp}.pth.tar")
logs_file =              join(logs_folder, f"Logs_{user_params_with_time_stamp}.txt")
predictions_file =       join(prediction_files_folder, f"Predictions_{user_params_with_time_stamp}.txt")
# endregion output files

def printF(s:str, fn=logs_file):
    print(s)
    open(fn, 'a+', encoding='utf8').write(s + '\n')

printF(user_params_config_to_print)
printF(f"""
Logs file: {logs_file}
Predictions file: {predictions_file}
Loss & Accuracy graph file: {evaluation_graphs_file}
Best model's folder: {model_checkpoints_folder}
""")

hyper_params_to_print = f"""#epochs = {hp.num_epochs},
lr = {hp.learning_rate},
batch = {hp.batch_size},
encoder_embed_size = {hp.encoder_embedding_size},
decoder_embed_size = {hp.decoder_embedding_size},
hidden_size = {hp.hidden_size},
time_stamp = {time_stamp}"""

printF(f"- Hyper-Params: {hyper_params_to_print}")
printF("- Defining a SummaryWriter object")
# use Tensorboard to get nice loss plot
hyper_params_to_print = hyper_params_to_print.replace("\n", " ") # more compact + needed later
summary_writer = SummaryWriter(summaryWriter_logs_folder, comment=hyper_params_to_print)
