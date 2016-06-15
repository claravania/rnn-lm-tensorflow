#!/usr/bin/python
# Author: Clara Vania

import numpy as np
import tensorflow as tf
import argparse
import time
import os
import codecs
import cPickle
from utils import TextLoader
from word import WordLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/tinyshakespeare/train.txt',
                        help="training data")
    parser.add_argument('--dev_file', type=str, default='data/tinyshakespeare/dev.txt',
                        help="development data")
    parser.add_argument('--output', '-o', type=str, default='train.log',
                        help='output file')
    parser.add_argument('--save_dir', type=str, default='model',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=200,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='minibatch size')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--out_vocab_size', type=int, default=10000,
                        help='size of output vocabulary')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='number of epochs')
    parser.add_argument('--validation_interval', type=int, default=1,
                        help='validation interval')
    parser.add_argument('--init_scale', type=float, default=0.1,
                        help='initial weight scale')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='maximum permissible norm of the gradient')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='the decay of the learning rate')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='the probability of keeping weights in the dropout layer')
    parser.add_argument('--optimization', type=str, default='sgd',
                        help='sgd, momentum, or adagrad')
    args = parser.parse_args()
    train(args)


def run_epoch(session, m, data, data_loader, eval_op, verbose=False):
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_lm_state.eval()
    for step, (x, y) in enumerate(data_loader.data_iterator(data, m.batch_size, m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_lm_state: state})
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def train(args):
    start = time.time()
    save_dir = args.save_dir
    try:
        os.stat(save_dir)
    except:
        os.mkdir(save_dir)

    with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    data_loader = TextLoader(args)
    train_data = data_loader.train_data
    dev_data = data_loader.dev_data

    out_file = os.path.join(args.save_dir, args.output)
    fout = codecs.open(out_file, "w", encoding="UTF-8")

    args.word_vocab_size = data_loader.word_vocab_size
    print "Word vocab size: " + str(data_loader.word_vocab_size) + "\n"
    fout.write("Word vocab size: " + str(data_loader.word_vocab_size) + "\n")

    # Model
    lm_model = WordLM

    print "Begin training..."
    # If using gpu:
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    # add parameters to the tf session -> tf.Session(config=gpu_config)
    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)

        # Build models
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtrain = lm_model(args, is_training=True)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mdev = lm_model(args, is_training=False)

        # save only the last model
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        tf.initialize_all_variables().run()
        dev_pp = 10000000.0

        # process each epoch
        e = 0
        decay_counter = 1
        learning_rate = args.learning_rate
        while e < args.num_epochs:
            if e > 4:
                lr_decay = args.decay_rate ** decay_counter
                learning_rate = args.learning_rate * lr_decay
                decay_counter += 1
            print("Epoch: %d" % (e + 1))
            mtrain.assign_lr(sess, learning_rate)
            print("Learning rate: %.3f" % sess.run(mtrain.lr))

            train_perplexity = run_epoch(sess, mtrain, train_data, data_loader, mtrain.train_op, verbose=True)
            print("Train Perplexity: %.3f" % train_perplexity)

            dev_perplexity = run_epoch(sess, mdev, dev_data, data_loader, tf.no_op())
            print("Valid Perplexity: %.3f" % dev_perplexity)

            # write results to file
            fout.write("Epoch: %d\n" % (e + 1))
            fout.write("Learning rate: %.3f\n" % sess.run(mtrain.lr))
            fout.write("Train Perplexity: %.3f\n" % train_perplexity)
            fout.write("Valid Perplexity: %.3f\n" % dev_perplexity)
            fout.flush()

            if dev_pp > dev_perplexity:
                print "Achieve highest perplexity on dev set, save model."
                checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e)
                print "model saved to {}".format(checkpoint_path)
                dev_pp = dev_perplexity
            e += 1

        print("Training time: %.0f" % (time.time() - start))
        fout.write("Training time: %.0f\n" % (time.time() - start))


if __name__ == '__main__':
    main()
