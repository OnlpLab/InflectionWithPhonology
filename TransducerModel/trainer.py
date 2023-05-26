import random
import time
from collections import Counter

import dynet as dy
from editdistance import eval as edit_distance_eval
import numpy as np
import progressbar

import util
from DataRelatedClasses.DataSets.BaseDataSet import BaseDataSet
from DataRelatedClasses.DataSamples.AlignedDataSample import AlignedDataSample
from DataRelatedClasses.utils import action2string
from defaults import SANITY_SIZE
from typing import Callable, List

OPTIMIZERS = {'ADAM'    : #dy.AdamTrainer,
                          lambda model: dy.AdamTrainer(model, alpha=0.0005, beta_1=0.9, beta_2=0.999, eps=1e-8),
              'MOMENTUM': dy.MomentumSGDTrainer,
              'SGD'     : dy.SimpleSGDTrainer,
              'ADAGRAD' : dy.AdagradTrainer,
              'ADADELTA': dy.AdadeltaTrainer}


def vote(outputs):
    outputs = [output for output in outputs if output]
    return Counter(outputs).most_common()[0][0]

# region eval_methods
def pcfp_internal_eval(batches, transducer, vocab,
                  previous_predicted_actions,
                  check_condition=False, name='test'):
    then = time.time()
    print('evaluating on {} data...'.format(name))

    number_correct = 0.
    total_loss = 0.
    predictions = []
    pred_acts = []
    i = 0  # counter of samples
    for j, batch in enumerate(batches):
        dy.renew_cg()
        batch_loss = []
        for multi_sample in batch:
            temp_predictions = []
            for sample in multi_sample.samples:
                in_feats = sample.in_pos, sample.in_feats
                out_feats = sample.out_pos, sample.out_feats
                loss, prediction, predicted_actions = transducer.transduce(sample.lemma, in_feats, out_feats, external_cg=True)
                temp_predictions.append(prediction)
            try:
                prediction = vote(temp_predictions)
            except IndexError:
                print(sample.lemma_str)
                prediction = ''
            predictions.append(prediction)
            pred_acts.append(predicted_actions)
            batch_loss.extend(loss)

            # evaluation
            correct_prediction = False
            if (prediction in vocab.word and vocab.word.w2i[prediction] == multi_sample.word):
                correct_prediction = True
                number_correct += 1

            # increment counter of samples
            i += 1
        batch_loss = -dy.average(batch_loss)
        total_loss += batch_loss.scalar_value()
        # report progress
        if j > 0 and j % 100 == 0: print('\t\t...{} batches'.format(j))

    accuracy = number_correct / i
    print('\t...finished in {:.3f} sec'.format(time.time() - then))
    return accuracy, total_loss, predictions, pred_acts


def internal_eval(batches, transducer, vocab, previous_predicted_actions, check_condition=True, name='train'):

    then = time.time()
    print('evaluating on {} data...'.format(name))

    number_correct = 0.
    total_loss = 0.
    predictions = []
    pred_acts = []
    edit_distances = []
    i = 0  # counter of samples
    for j, batch in enumerate(batches):
        dy.renew_cg()
        batch_loss = []
        for sample in batch:
            in_feats = sample.in_pos, sample.in_feats
            out_feats = sample.out_pos, sample.out_feats
            loss, prediction, predicted_actions = transducer.transduce(sample.lemma, in_feats, out_feats,
                                                                       external_cg=True, phonemes=sample.phonemes)
            predictions.append(prediction)
            pred_acts.append(predicted_actions)
            batch_loss.extend(loss)

            # evaluation
            correct_prediction = False
            if prediction in vocab.word and vocab.word.w2i[prediction] == sample.word:
                correct_prediction = True
                number_correct += 1
                edit_distances.append(0.0)
            else:
                gold_target = sample.phonemes_str if sample.phonemes_str else sample.word_str
                edit_distances.append(edit_distance_eval(gold_target, prediction))

            if check_condition:
                # display prediction for this sample if it differs the prediction
                # of the previous epoch or its an error
                if predicted_actions != previous_predicted_actions[i] or not correct_prediction:
                    #
                    print('BEFORE:    ', action2string(previous_predicted_actions[i], vocab))
                    print('THIS TIME: ', action2string(predicted_actions, vocab))
                    print('TRUE:      ', sample.act_repr)
                    print('PRED:      ', prediction)
                    print('WORD:      ', sample.word_str)
                    print('X' if correct_prediction else 'V')
            # increment counter of samples
            i += 1
        batch_loss = -dy.average(batch_loss)
        total_loss += batch_loss.scalar_value()
        # report progress
        if j > 0 and j % 100 == 0: print('\t\t...{} batches'.format(j))

    accuracy = number_correct / i
    edit_distance = np.mean(edit_distances)
    print(f'\t...finished in {(time.time() - then):.3f} sec')
    return accuracy, total_loss, predictions, pred_acts, edit_distance


def internal_eval_beam(batches, transducer, vocab,
                  beam_width, previous_predicted_actions,
                  check_condition=True, name='train'):
    assert callable(getattr(transducer, "beam_search_decode", None)), 'transducer does not implement beam search.'
    then = time.time()
    print('evaluating on {} data with beam search (beam width {})...'.format(name, beam_width))
    number_correct = 0.
    total_loss = 0.
    predictions = []
    pred_acts = []
    i = 0  # counter of samples
    for j, batch in enumerate(batches):
        dy.renew_cg()
        batch_loss = []
        for multi_sample in batch:
            temp_predictions = []
            for sample in multi_sample.samples:
                in_feats = sample.in_pos, sample.in_feats
                out_feats = sample.out_pos, sample.out_feats
                hypotheses = transducer.beam_search_decode(sample.lemma, in_feats, out_feats, external_cg=True,
                                                            beam_width=beam_width)
                # take top hypothesis
                try:
                    loss, loss_expr, prediction, predicted_actions = hypotheses[0]
                except Exception as e:
                    print(hypotheses)
                    raise e
                temp_predictions.append(prediction)
            try:
                prediction = vote(temp_predictions)
            except IndexError:
                print(sample.lemma_str)
                prediction = ''
            predictions.append(prediction)
            pred_acts.append(predicted_actions)
            batch_loss.append(loss)
            # sanity check: Basically, this often is wrong...
            #assert round(loss, 3) == round(loss_expr.scalar_value(), 3), (loss, loss_expr.scalar_value())

            # evaluation
            correct_prediction = False
            if (prediction in vocab.word and vocab.word.w2i[prediction] == sample.word):
                correct_prediction = True
                number_correct += 1
                if check_condition:
                    # compare to greedy prediction:
                    _, greedy_prediction, _ = transducer.transduce(sample.lemma, feats, external_cg=True)
                    if greedy_prediction != prediction:
                        print('Beam! Target: ', sample.word_str)
                        print('Greedy prediction: ', greedy_prediction)
                        print('Complete hypotheses:')
                        for log_p, _, pred_word, pred_actions in hypotheses:
                            print('Actions {}, word {}, -log p {:.3f}'.format(
                                action2string(pred_actions, VOCAB), pred_word, -log_p))

            if check_condition:
                # display prediction for this sample if it differs the prediction
                # of the previous epoch or its an error
                if predicted_actions != previous_predicted_actions[i] or not correct_prediction:
                    #
                    print('BEFORE:    ', action2string(previous_predicted_actions[i], vocab))
                    print('THIS TIME: ', action2string(predicted_actions, vocab))
                    print('TRUE:      ', sample.act_repr)
                    print('PRED:      ', prediction)
                    print('WORD:      ', sample.word_str)
                    print('X' if correct_prediction else 'V')
            # increment counter of samples
            i += 1
        batch_loss = -np.mean(batch_loss)
        total_loss += batch_loss
        # report progress
        if j > 0 and j % 100 == 0: print('\t\t...{} batches'.format(j))

    accuracy = number_correct / i
    print('\t...finished in {:.3f} sec'.format(time.time() - then))
    return accuracy, total_loss, predictions, pred_acts
# endregion eval_methods

class TrainingSession(object):
    def __init__(self, model, transducer, vocab,
                 train_data: BaseDataSet, dev_data: BaseDataSet,
                 batch_size,
                 optimizer=None,
                 decbatch_size=None,
                 dev_batches=None):

        self.model = model
        self.transducer = transducer
        self.optimizer = OPTIMIZERS.get(optimizer, 'ADADELTA')
        self.trainer = None  # initialized only in training
        self.vocab = vocab

        # DATA and BATCHES
        self.train_data = train_data
        self.dev_data = dev_data
        self.dev_batches = dev_batches

        self.batch_size = batch_size
        # use different (larger) batch size for decoding
        self.decbatch_size = decbatch_size if decbatch_size else batch_size
        self.dev_len    = len(self.dev_data) if self.dev_data else 0
        self.train_len  = len(self.train_data)
        if self.dev_batches is None:
            self.dev_batches = [self.dev_data.samples[i:i+self.decbatch_size]
                for i in range(0, self.dev_len, self.decbatch_size)]

        sanity_size = min(SANITY_SIZE, len(self.train_data))
        self.sanity_batches = [self.train_data.samples[:sanity_size][i:i+self.decbatch_size]
            for i in range(0, sanity_size, self.decbatch_size)]

        print('Decoding batch size is {}.'.format(self.decbatch_size))
        print('Training batch size is {}.'.format(self.batch_size))
        print('There are {} train and {} dev samples.'.format(self.train_len, self.dev_len))
        print('There are {} train batches and {} dev batches.'.format(
            (self.train_len / self.batch_size) + 1, len(self.dev_batches)))

        # BOOKKEEPING OF PREDICTED ACTIONS
        self.dev_predicted_actions = [None]*self.dev_len
        self.train_predicted_actions = [None]*sanity_size

        # PERFORMANCE METRICS
        # dev performance stats
        self.best_avg_dev_loss = 999.
        self.best_dev_accuracy = 0.
        self.best_dev_loss_epoch = 0
        self.best_dev_acc_epoch  = 0
        # train performance stats
        self.avg_loss = 0.
        self.best_train_accuracy = 0.

    def reload(self, path2model, tmp_model_path=None):
        print('Trying to reload model from: {}'.format(path2model))
        self.model.populate(path2model)
        print('Computing dev accuracy of the reloaded model...')
        # initialize dev stats from the pretrained model
        # self.best_dev_accuracy, self.best_avg_dev_loss = \
        #     self.dev_eval(check_condition=False)
        # print 'Dev accuracy, dev loss: ', self.best_dev_accuracy, self.best_avg_dev_loss
        self.best_dev_loss_epoch = -1
        self.best_dev_acc_epoch  = -1
        if tmp_model_path and tmp_model_path != path2model:
            self.model.save(tmp_model_path)
            print('saved reloaded model as best model to {}'.format(tmp_model_path))

    def action2string(self, acts):
        return action2string(acts, self.vocab)

    def dev_eval(self, check_condition=True):
        # call internal_eval with dev batches
        dev_accuracy, avg_dev_loss, _, self.dev_predicted_actions, dev_edit_distance = \
            internal_eval(self.dev_batches, self.transducer, self.vocab,
                          self.dev_predicted_actions,
                          check_condition=check_condition, name='dev')
        return dev_accuracy, avg_dev_loss, dev_edit_distance

    def train_eval(self, check_condition=True):
        # call internal_eval with train batches
        train_dev_accuracy, avg_loss, _, self.train_predicted_actions, _ = \
            internal_eval(self.sanity_batches, self.transducer, self.vocab,
                          self.train_predicted_actions,
                          check_condition=check_condition, name='train')
        return train_dev_accuracy, avg_loss

    def run_MLE_training(self, **kwargs):

        print('Running MLE training...')
        l2 = kwargs.get('l2')
        if l2:
            print('Using l2-regularization with parameter {}'.format(l2))

        self.model.save(kwargs['tmp_model_path'])
        print('saved initial model to {}'.format(kwargs['tmp_model_path']))

        def MLE_batch_update(batch: List[AlignedDataSample], *args):
            # How to update model parameters from
            # a batch of training samples with MLE?
            dy.renew_cg()
            batch_loss = []
            for sample in batch:
                in_feats = sample.in_pos, sample.in_feats
                out_feats = sample.out_pos, sample.out_feats
                loss, _, _ = self.transducer.transduce(sample.lemma, in_feats, out_feats, sample.actions, external_cg=True, phonemes=sample.phonemes)
                batch_loss.extend(loss)
            batch_loss = -dy.average(batch_loss)
            if l2: batch_loss += l2 * self.transducer.l2_norm(with_embeddings=False)
            loss = batch_loss.scalar_value()  # forward
            batch_loss.backward()             # backward
            self.trainer.update()
            return loss

        self.run_training(MLE_batch_update, **kwargs)


    def run_RL_training(self, **kwargs):

        print('Running RL training...')
        #print 'Trainer attributes: ', self.trainer.__dict__

        sample_size = kwargs['sample_size']
        scale_neg = kwargs['scale_negative']
        beta_ned = kwargs['beta']
        use_baseline = kwargs['baseline']
        verbose = True if kwargs['check_condition'] else False

        print('Will draw {} samples per training sample.'.format(sample_size))
        print('Will use greedy baseline for reward correction.' if use_baseline else 'Will not use baseline reward correction.')
        print('Will apply negative scaling of {}.'.format(scale_neg))

        def compute_reward(word, word_str, prediction):
            # `word` is an integer code,
            # `word_str` is the string corresponding to this code,
            # `prediction` is a string
            if (prediction in self.vocab.word and
                self.vocab.word.w2i[prediction] == word):
                # correct prediction
                reward = 1.
            else:
                # the smaller the distance the better
                #reward = -1*int(edit_distance_eval(word_str, prediction))/len(word_str)
                reward = -beta_ned * edit_distance_eval(word_str, prediction) / max(len(word_str), len(prediction))
            return reward

        def RL_batch_update(batch, *args):

            dy.renew_cg()
            batch_loss = []
            rewards = []
            for sample in batch:

                lemma = sample.lemma
                word = sample.word
                word_str = sample.word_str
                feats = sample.pos, sample.feats

                if use_baseline:
                    # BASELINE PREDICTION
                    _, prediction_b, predicted_actions_b = \
                        self.transducer.transduce(lemma, feats, external_cg=True)
                    # BASELINE REWARD
                    reward_b = compute_reward(word, word_str, prediction_b)
                else:
                    actions = tuple(sample.actions)

                for _ in range(sample_size):

                    # SAMPLING-BASED PREDICTION
                    loss, prediction, predicted_actions = \
                        self.transducer.transduce(lemma, feats, sampling=True, external_cg=True)
                    # SAMPLING-BASED REWARD
                    reward = compute_reward(word, word_str, prediction)

                    if use_baseline:
                        sample_reward = reward - reward_b
                    else:
                        sample_reward = 0. if tuple(predicted_actions) == actions else reward

                    if verbose and use_baseline and sample_reward and reward == 1.:
                        # i.e. sampling produced a correct prediction via a sequence of actions
                        # different from the argmax approach of the baseline.
                        assert predicted_actions != predicted_actions_b
                        print(('Correct prediction by sampling for {}, {}:\n'
                                'Sampling: {}\t{}\n'
                                'Baseline: {}\t{}\n'.format(
                            sample.lemma_str, sample.feat_str,
                            prediction, self.action2string(predicted_actions),
                            prediction_b, self.action2string(predicted_actions_b))))

                    if scale_neg and sample_reward < 0:
                        sample_reward = scale_neg*sample_reward

                    if sample_reward:
                        rewards.append(sample_reward)
                        batch_loss.append(-dy.average(loss))
                    #print 'word, prediction_b, prediction: ', word_str, prediction_b, prediction
                    #print 'reward_b, reward, sample_reward: ', reward_b, reward, sample_reward
                    print('word, prediction: ', word_str, prediction)
                    print('reward, sample_reward: ', reward, sample_reward)
            if batch_loss:
                num_nonzero_grad = len(batch_loss)
                # dy.concatenate(batch_loss) => make a vector out of Python list of dynet scalars
                # dy.cdiv => element-wise division, then .scalar_value() to get a scalar
                # division is not implemented.
                batch_loss = dy.cdiv(dy.dot_product(dy.inputVector(rewards),
                    dy.concatenate(batch_loss)), dy.scalarInput(num_nonzero_grad))
                loss = batch_loss.scalar_value()  # forward
                batch_loss.backward()
                self.trainer.update()
                if verbose:
                    print('Batch loss, batch reward: ', loss, sum(rewards)/num_nonzero_grad)
            else:
                loss = 0
                if verbose:
                    print('Batch loss is zero.')
            return loss

        self.run_training(RL_batch_update, **kwargs)


    def run_MRT_training(self, **kwargs):

        print('Running MRT training with sampling...')
        #print 'Trainer attributes: ', self.trainer.__dict__

        sample_size = kwargs['sample_size']
        alpha_p  = kwargs['alpha']  #0.05
        beta_ned = kwargs['beta']
        verbose = True if kwargs['check_condition'] else False

        print('Alpha parameter will be {}'.format(alpha_p))
        print('Beta scaling factor for NED will be {}'.format(beta_ned))
        print('Sample size will be {}'.format(sample_size))

        def compute_reward(word, word_str, prediction):
            # `word` is an integer code,
            # `word_str` is the string corresponding to this code,
            # `prediction` is a string

            # This is a normalized edit distance cost
            # The better the prediction, the lower the reward.
            if (prediction in self.vocab.word and
                self.vocab.word.w2i[prediction] == word):
                # correct prediction
                reward = -1.
            else:
                # the smaller the distance the better
                reward = beta_ned * edit_distance_eval(word_str, prediction) / max(len(word_str), len(prediction))
            return reward

        def MRT_batch_update(batch, epoch):

            dy.renew_cg()

            alpha = dy.scalarInput(alpha_p)

            batch_loss = []
            rewards = []
            for sample in batch:

                lemma = sample.lemma
                word = sample.word
                word_str = sample.word_str
                in_feats = sample.in_pos, sample.in_feats
                out_feats = sample.out_pos, sample.out_feats
                actions = sample.actions

                # ORACLE PREDICTION
                #loss, prediction_b, predicted_actions_b = \
                gold_loss, _, _ = \
                    self.transducer.transduce(lemma, in_feats, out_feats, actions, external_cg=True)
                gold_loss = dy.esum(gold_loss)
                #if gold_loss.scalar_value() < -50.:  # Sum log P
                #    print 'Dangerously low prob of gold action seq: ', gold_loss.scalar_value(), word_str
                #    hypotheses = []
                #else:
                #    hypotheses = [ (_, gold_loss, word_str, actions) ]

                # BEAM-SEARCH-BASED PREDICTION
                #hypotheses += self.transducer.beam_search_decode(lemma, feats, external_cg=True,
                #                                                 beam_width=beam_width)
                sample_rewards = [-1.]
                sample_losses = [gold_loss]
                predictions = [word_str]
                seen_predicted_acts = {tuple(actions)}
                #for _, loss, prediction, predicted_actions in hypotheses:
                for _ in range(sample_size):
                    loss, prediction, predicted_actions = \
                        self.transducer.transduce(lemma, in_feats, out_feats, sampling=True, external_cg=True)
                    predicted_actions = tuple(predicted_actions)
                    if predicted_actions in seen_predicted_acts:
                        #if verbose: print 'already sampled this action sequence: ', predicted_actions
                        continue
                    loss = dy.esum(loss)
                    if loss.scalar_value() < -20:  # log P
                        continue
                    else:
                        seen_predicted_acts.add(predicted_actions)
                #for _, loss, prediction, predicted_actions in hypotheses:

                    # COMPUTE REWARDS
                    reward = compute_reward(word, word_str, prediction)

                    sample_rewards.append(reward)
                    sample_losses.append(loss)
                    predictions.append(prediction)

                # SCALE & RENORMALIZE: (these are log P)
                if len(sample_rewards) == 1 and sample_rewards[0] == -1.:
                    if verbose: print('Nothing to update with.')
                    continue
                else:
                    #if verbose: print 'sample_losses', sample_losses
                    sample_losses = dy.concatenate(sample_losses)
                    sample_rewards = dy.inputVector(sample_rewards)
                    q_unnorm = dy.pow(dy.exp(sample_losses), alpha)
                    q = dy.cdiv(q_unnorm, dy.sum_elems(q_unnorm))

                    if verbose:
                        print('q', q.npvalue())
                        print('sample_rewards', sample_rewards.npvalue())
                        print('word', word_str)
                        print('predictions: ', ', '.join(predictions))
                    batch_loss.append(dy.dot_product(q, sample_rewards))
            if batch_loss:
                batch_loss = dy.esum(batch_loss)
                loss = batch_loss.scalar_value()  # forward
                try:
                    batch_loss.backward()
                    self.trainer.update()
                except Exception as e:
                    print('Batch loss: ', loss)
                    print('q', q.npvalue())
                    print('q_unnorm', q_unnorm.npvalue())
                    print('gold_loss', gold_loss.scalar_value())
                    print('sample_rewards', sample_rewards.npvalue())
                    print('word', word_str)
                    print('predictions: ', ', '.join(predictions))
                    raise e
                if verbose: print('Batch loss: ', loss)
            else:
                if verbose: print('Batch loss is zero.')
                loss = 0.
            return loss

        self.run_training(MRT_batch_update, **kwargs)


    def run_training(self, batch_update, epochs, max_patience, pick_best_accuracy, dropout, log_file_path,
                     tmp_model_path, check_condition, train_until_accuracy=None, optimizer=None, **kwargs):
        if optimizer is None:
            optimizer = self.optimizer
        self.trainer = optimizer(self.model)
        print('Initialized trainer with: {}.'.format(optimizer))

        if dropout:
            print('Using dropout of {}.'.format(dropout))
        else:
            print('Not using dropout.')

        if check_condition is False:
            check_condition = lambda e: False
        elif check_condition == 2:  # max verbose flag...
            check_condition = lambda e: e > 0

        if train_until_accuracy and 0 < train_until_accuracy <= 1.:
            epochs = 10000
            max_patience = 10000
            print('Will train until training set accuracy of {} is reached.'.format(train_until_accuracy))
        else:
            print('Will train for a maximum of {} epochs with patience of {}.'.format(epochs, max_patience))
        print('Will early stop based on dev {}.'.format('accuracy' if pick_best_accuracy else 'loss'))

        # PROGRESS BAR INIT
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()

        # LOG FILE INIT
        with open(log_file_path, 'a') as a:
            a.write('epoch\tavg_loss\ttrain_accuracy\tdev_accuracy\tdev_edit_distance\n')

        patience = 0
        for epoch in range(epochs):
            print('training...')
            then = time.time()
            train_loss = 0.

            train = self.train_data.samples
            random.shuffle(train)
            batches = [train[i:i+self.batch_size] for i in range(0, self.train_len, self.batch_size)]
            print('Number of train batches: {}.'.format(len(batches)))

            # ENABLE DROPOUT
            if dropout: self.transducer.set_dropout(dropout)

            for j, batch in enumerate(batches):
                train_loss += batch_update(batch, epoch)
                if j > 0 and j % 100 == 0: print('\t\t...{} batches'.format(j))
            print(f'\t\t...{j} batches')

            # DISABLE DROPOUT AFTER TRAINING
            if dropout: self.transducer.disable_dropout()
            print(f'\t...finished in {(time.time() - then):.3f} sec')
            self.avg_loss = train_loss / self.train_len
            print('Average train loss: ', self.avg_loss)

            # EVALUATE MODEL ON SUBSET OF TRAIN (SANITY)
            train_accuracy, avg_loss = self.train_eval(check_condition(epoch))
            if train_accuracy > self.best_train_accuracy:
                self.best_train_accuracy = train_accuracy

            patience += 1

            # EVALUATE MODEL ON DEV
            if self.dev_data:
                dev_accuracy, avg_dev_loss, dev_edit_distance = self.dev_eval(check_condition(epoch))

                if dev_accuracy > self.best_dev_accuracy:
                    self.best_dev_accuracy = dev_accuracy
                    self.best_dev_acc_epoch = epoch
                    # using dev acc for early stopping
                    print(f'Found best dev accuracy so far {self.best_dev_accuracy:.7f}')
                    if pick_best_accuracy: patience = 0

                if avg_dev_loss < self.best_avg_dev_loss:
                    self.best_avg_dev_loss = avg_dev_loss
                    self.best_dev_loss_epoch = epoch
                    # using dev loss for early stopping
                    print(f'Found best dev loss so far {self.best_avg_dev_loss:.7f}')
                    if not pick_best_accuracy: patience = 0

                if patience == 0:
                    # patience has been reset to 0, so save currently best model
                    self.model.save(tmp_model_path)
                    print('saved new best model to {}'.format(tmp_model_path))

                print(f'epoch: {epoch}, train acc: {train_accuracy:.4f}, best train acc: {self.best_train_accuracy:.4f}, '
                      f'train loss: {self.avg_loss:.4f}, dev acc: {dev_accuracy:.4f}, best dev acc: {self.best_dev_accuracy:.4f}'
                      f' (epoch {self.best_dev_acc_epoch}), dev loss: {avg_dev_loss:.4f}, best dev loss: {self.best_avg_dev_loss:.7f}'
                      f' (epoch {self.best_dev_loss_epoch}), patience = {patience}')

            else:
                dev_accuracy = -1
                dev_edit_distance = -1
                patience = 0
                self.model.save(tmp_model_path)
                print('saved last model to {}'.format(tmp_model_path))

                print(f'epoch: {epoch}, train acc: {train_accuracy:.4f}, best train acc: {self.best_train_accuracy:.4f}, '
                      f'train loss: {self.avg_loss:.4f}, best dev acc: {self.best_dev_accuracy:.4f} (epoch {self.best_dev_acc_epoch}), '
                      f'best dev loss: {self.best_avg_dev_loss:.7f} (epoch {self.best_dev_loss_epoch}), patience = {patience}')

            # LOG LATEST RESULTS
            with open(log_file_path, 'a') as a:
                a.write(f"{epoch}\t{self.avg_loss:.6f}\t{train_accuracy}\t{dev_accuracy}\t{dev_edit_distance}\n")

            if patience == max_patience:
                print('out of patience after {} epochs'.format(epoch + 1))
                train_progress_bar.finish()
                break
            if train_until_accuracy and train_accuracy > train_until_accuracy:
                print('reached required training accuracy level of {}'.format(train_until_accuracy))
                train_progress_bar.finish()
                break

            # UPDATE PROGRESS BAR
            train_progress_bar.update(epoch)


def withheld_data_eval(name, batches, transducer, vocab, beam_widths,
                       pred_path: Callable[[str], str], gold_path, sigm2017format):

    """Runs internal greedy and beam-search evaluations as well as
       launches external eval script. Returns greedy accuracy (hm...?)"""

    # GREEDY PREDICTIONS FROM THIS MODEL 
    greedy_accuracy, _, predictions, _, _ = internal_eval(batches,
        transducer, vocab, None, check_condition=False, name=name)
    if greedy_accuracy > 0:
        print('{} accuracy: {}'.format(name, greedy_accuracy))
    else:
        print('Possibly covered test data. Accuracy zero.')
    # write out greedy predictions and scores
    util.external_eval(pred_path('greedy'), gold_path, batches, predictions, sigm2017format)

    # BEAM-SEARCH-BASED PREDICTIONS FROM THIS MODEL
    if beam_widths:
        print('\nDecoding with beam search...')
        from Models import hacm_sub
        if not callable(getattr(transducer, "beam_search_decode", None)) or \
            isinstance(transducer, hacm_sub.MinimalTransducer):
            print('Transducer does not implement beam search.')
        else:
            for beam_width in beam_widths:
                accuracy, _, predictions, _ = internal_eval_beam(batches,
                    transducer, vocab, beam_width, None, check_condition=False, name=name)
                if accuracy > 0:
                    print('beam-{} accuracy {}'.format(beam_width, accuracy))
                else:
                    print('Zero accuracy.')
                # write out predictions and scores more specifically
                beam_path = pred_path('beam' + str(beam_width))
                util.external_eval(beam_path, gold_path, batches, predictions, sigm2017format)
        print('all in all: {} greedy accuracy, {} beam accuracy'.format(greedy_accuracy, accuracy))

    return greedy_accuracy


def dev_external_eval(batches, transducer, vocab, paths, data_arguments, model_arguments, optim_arguments):
    accuracy =  withheld_data_eval("dev", batches, transducer, vocab, optim_arguments['beam-widths'],
                       paths['dev_output'], paths['dev_path'], data_arguments['sigm2017format'])
    # WRITE STATS TO FILE (NOT IN TEST EXTERNAL EVAL)
    util.write_stats_file(accuracy, paths, data_arguments, model_arguments, optim_arguments)


def test_external_eval(batches, transducer, vocab, paths, beam_widths, sigm2017format):

    accuracy = withheld_data_eval("test", batches, transducer, vocab, beam_widths,
                                  paths['test_output'], paths['test_path'], sigm2017format)
    with open(paths['stats_file_path'], 'a+') as f:
        f.write(f"TEST ACCURACY (internal evaluation) = {accuracy}\n")

    return accuracy