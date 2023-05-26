import random, os, time
from datetime import timedelta
from torch import manual_seed, load
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torchtext.legacy.data import BucketIterator, TabularDataset
from editdistance import eval as edit_distance_eval

import hyper_params_config as hp
from run_setup import train_file, dev_file, test_file, model_checkpoints_folder, model_checkpoint_file, predictions_file, \
    hyper_params_to_print, summary_writer, evaluation_graphs_file, get_time_now_str, user_params_with_time_stamp, printF
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint, srcField, trgField, device, plt
from phonology_decorator import phonology_decorator, is_features_bundle
from network import Encoder, Decoder, Seq2Seq

def show_readable_triplet(src, trg, pred):
    # Presents the triplet in a more tidy way (no converting)
    src_print = [e.replace(',',';' if is_features_bundle(e) else ',' if hp.inp_phon_type=='features' else '') for e in ','.join(src).split(',+,')]
    trg_print, pred_print = (','.join(trg), ','.join(pred)) if hp.out_phon_type=='features' else (''.join(trg), ''.join(pred))
    return src_print, trg_print, pred_print

def save_plots_figures(EDs_phon, accs_phon, EDs_morphs, accs_morph):
    plt.figure(figsize=(10,8)), plt.suptitle(f'Development Set Results, {hp.training_mode}-split')
    if hp.PHON_REEVALUATE:
        plt.subplot(221), plt.title("Phon-ED"), plt.plot(EDs_phon)
        plt.subplot(222), plt.title("Phon-Acc"), plt.plot(accs_phon)
        plt.subplot(223), plt.title("Morph-ED"), plt.plot(EDs_morphs)
        plt.subplot(224), plt.title("Morph-Acc"), plt.plot(accs_morph)
    else:
        plt.subplot(121), plt.title("Morph-Acc"), plt.plot(accs_morph)
        plt.subplot(122), plt.title("Morph-ED"), plt.plot(EDs_morphs)
    printF(f'Saving the plot of the results on {evaluation_graphs_file}')
    plt.savefig(evaluation_graphs_file)

def main():
    # Note: the arguments parsing occurs globally at hyper_params_config.py
    t0=time.time()
    random.seed(hp.SEED)
    manual_seed(hp.SEED) # torch.manual_seed
    save_model = True
    summary_writer_step = 0

    printF("- Generating the datasets:")
    printF(f"\ttrain_file = {train_file}, dev_file = {dev_file}")
    train_data, dev_data, test_data = TabularDataset.splits(path='', train=train_file, validation=dev_file, test=test_file,
                             fields=[("src", srcField), ("trg", trgField)], format='tsv') # test data is out of the game.
    printF("- Building vocabularies")
    srcField.build_vocab(train_data, dev_data) # no limitation of max_size or min_freq is needed.
    trgField.build_vocab(train_data, dev_data) # no limitation of max_size or min_freq is needed.

    printF("- Generating BucketIterator objects")
    train_iterator, dev_iterator = BucketIterator.splits(
        (train_data, dev_data),
        batch_size=hp.batch_size,
        sort_within_batch=True,
        sort_key= lambda x: len(x.src),
        device=device
    )

    input_size_encoder = len(srcField.vocab)
    input_size_decoder = len(trgField.vocab)
    output_size = len(trgField.vocab)

    # region defineNets
    printF("- Constructing networks & optimizer")
    encoder_net = Encoder(input_size_encoder, hp.encoder_embedding_size, hp.hidden_size, hp.num_layers, hp.enc_dropout).to(device)
    decoder_net = Decoder(input_size_decoder, hp.decoder_embedding_size, hp.hidden_size, output_size, hp.num_layers, hp.dec_dropout,).to(device)
    model = Seq2Seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)

    printF("- Defining some more stuff...")
    criterion = CrossEntropyLoss(ignore_index=srcField.vocab.stoi["<pad>"]) # '<pad>''s index
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=hp.LR_patience, verbose=True, factor=hp.LR_factor) # mode='max' bc we want to maximize the accuracy
    # endregion defineNets

    indices = random.sample(range(len(dev_data)), k=10)
    examples_for_printing = [dev_data.examples[i] for i in indices] # For a more interpretable evaluation, we apply translate_sentence on 10 samples.
    accs_phon, EDs_phon, accs_morph, EDs_morphs, best_measures = [], [], [], [], []
    max_morph_acc, ED_phon, accuracy_phon, ED_morph, accuracy_morph = -0.001, -0.001, -0.001, -0.001, -0.001

    printF("Let's begin training!\n")
    for epoch in range(1, hp.num_epochs + 1):
        printF(f"[Epoch {epoch} / {hp.num_epochs}] (hyper-params: {hyper_params_to_print})")
        printF(f"lr = {optimizer.state_dict()['param_groups'][0]['lr']:.7f}")
        model.train()
        printF(f"Starting the epoch on: {get_time_now_str(allow_colon=True)}.")
        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # Forward prop
            output = model(inp_data, target)

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin. While we're at it
            # Let's also remove the start token while we're at it
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # Back prop
            loss.backward()

            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # Plot to tensorboard
            summary_writer.add_scalar("Training loss", loss, global_step=summary_writer_step)
            summary_writer_step += 1

        model.eval()
        # Evaluate the performances on examples_for_printing
        for i,ex in enumerate(examples_for_printing, start=1):
            translated_sent = translate_sentence(model, ex.src, srcField, trgField, device, max_length=50)
            if translated_sent[-1]=='<eos>':
                translated_sent = translated_sent[:-1]
            src, trg, pred = ex.src, ex.trg, translated_sent # all the outputs are [str]; represents phonological stuff only if hp.out_phon_type!='graphemes'
            phon_ed = edit_distance_eval(trg, pred)
            src_print, trg_print, pred_print = show_readable_triplet(src, trg, pred)
            printF(f"{i}. input: {src_print} ; gold: {trg_print} ; pred: {pred_print} ; ED = {phon_ed}")

            # The next steps:
            # 1. If needed, convert the samples to a readable format (graphemes). Handle separately sources and trgs-preds.
            # 2. If needed, supply another evaluation of the prediction (for now, only graphemes-evaluation)
            # 3. Print the results. Refer to whether 1. and 2. were applied.

            # Convert non-graphemic formats to words
            if hp.inp_phon_type!='graphemes' or hp.out_phon_type!='graphemes':
                src_morph, trg_morph, pred_morph = phonology_decorator.phon_sample2morph_sample(src, trg, pred)
                if hp.out_phon_type!='graphemes': # another evaluation metric is needed; the source format is irrelevant
                    morph_ed_print = edit_distance_eval(trg_morph, pred_morph)
                    printF(f"{i}. input_morph: {src_morph} ; gold_morph: {trg_morph} ; pred_morph: {pred_morph} ; morphlvl_ED = {morph_ed_print}\n")
                else:
                    printF(f"{i}. input_morph: {src_morph} ; gold_morph: {''.join(trg_morph)} ; pred_morph: {''.join(pred_morph)}\n")


        if hp.PHON_REEVALUATE:
            ED_phon, accuracy_phon, ED_morph, accuracy_morph = bleu(dev_data, model, srcField, trgField, device, converter=phonology_decorator, output_file=predictions_file)
            summary_writer.add_scalar("Dev set Phon-Accuracy", accuracy_phon, global_step=epoch)
            extra_str = f"; avgED_phon = {ED_phon}; avgAcc_phon = {accuracy_phon}"
            accs_phon.append(accuracy_phon)
            EDs_phon.append(ED_phon)
        else:
            ED_morph, accuracy_morph = bleu(dev_data, model, srcField, trgField, device)
            extra_str=''
        summary_writer.add_scalar("Dev set Morph-Accuracy", accuracy_morph, global_step=epoch)
        printF(f"avgEDmorph = {ED_morph}; avgAcc_morph = {accuracy_morph}{extra_str}\n")

        accs_morph.append(accuracy_morph)
        EDs_morphs.append(ED_morph)
        printF(f"Ending the epoch on: {get_time_now_str(allow_colon=True)}.")

        # region model_selection
        if save_model:
            if epoch == 1: # first epoch
                save_checkpoint(model, optimizer, filename=model_checkpoint_file.replace('Model_Checkpoint',f'Model_Checkpoint_1'))
            else:
                # Check whether the last morph_accuracy was higher than the max. If yes, replace the ckpt with the last one.
                if accuracy_morph > max_morph_acc:
                    max_morph_acc = accuracy_morph
                    best_measures = [ED_phon, accuracy_phon, ED_morph, accuracy_morph, epoch] if hp.PHON_REEVALUATE else [ED_morph, accuracy_morph, epoch]
                    assert len([f for f in os.listdir(model_checkpoints_folder) if user_params_with_time_stamp in f]) == 1
                    ckpt_to_delete = [os.path.join(model_checkpoints_folder, f) for f in os.listdir(model_checkpoints_folder) if user_params_with_time_stamp in f][0]
                    temp_ckpt_name = model_checkpoint_file.replace('Model_Checkpoint',f'Model_Checkpoint_{epoch}')
                    save_checkpoint(model, optimizer, filename=temp_ckpt_name, file_to_delete=ckpt_to_delete)
                else: printF(f"Checkpoint didn't change. Current best (Accuracy={max_morph_acc}) achieved at epoch {best_measures[-1]}")
        # endregion model_selection
        lr_scheduler.step(accuracy_morph) # update only after model_selection

    # Load the best checkpoint and apply it on the dev set one last time. Report the results and make sure they are equal to best_measures.
    printF("Loading the best model")
    best_model_checkpoint_file = [os.path.join(model_checkpoints_folder, f) for f in os.listdir(model_checkpoints_folder) if user_params_with_time_stamp in f][0]
    load_checkpoint(load(best_model_checkpoint_file), model, optimizer)

    for test_set in [dev_data, test_data]:
        printF(f"Applying model on {'dev' if test_set==dev_data else 'test'} set")
        # test_set = test_data
        if hp.PHON_REEVALUATE:
            ED_phon, accuracy_phon, ED_morph, accuracy_morph = bleu(test_set, model, srcField, trgField, device, converter=phonology_decorator, output_file=predictions_file)
            if test_set == dev_data: assert [ED_phon, accuracy_phon, ED_morph, accuracy_morph] == best_measures[:-1] # sanity check
            printF(f"Phonological level: ED score on dev set is {ED_phon}. Avg-Accuracy is {accuracy_phon}.")
        else:
            ED_morph, accuracy_morph = bleu(test_set, model, srcField, trgField, device, output_file=predictions_file)
            if test_set == dev_data: assert [ED_morph, accuracy_morph] == best_measures[:-1] # sanity check, for debugging purposes
        printF(f"Morphological level: ED = {ED_morph}, Avg-Accuracy = {accuracy_morph}.")

    save_plots_figures(EDs_phon, accs_phon, EDs_morphs, accs_morph)

    printF(f'Elapsed time is {str(timedelta(seconds=time.time()-t0))}. Goodbye!')

if __name__ == '__main__':
    main()