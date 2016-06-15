#!/usr/bin/python
# Author: Clara Vania

import numpy as np
import tensorflow as tf
import argparse
import time
import os
import cPickle
from utils import TextLoader
from c2w import C2WLM
from s2w import S2WLM
from word import WordLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/tutorial/test.txt',
                        help="test file")
    parser.add_argument('--save_dir', type=str, default='model',
                        help='directory of the checkpointed models')
    args = parser.parse_args()
    test(args)


def run_epoch(session, m, data, data_loader, eval_op):
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

    return np.exp(costs / iters)


def test(test_args):
    start = time.time()
    with open(os.path.join(test_args.save_dir, 'config.pkl')) as f:
        args = cPickle.load(f)
    data_loader = TextLoader(args, train=False)
    test_data = data_loader.read_dataset(test_args.test_file)

    args.word_vocab_size = data_loader.word_vocab_size
    print "Word vocab size: " + str(data_loader.word_vocab_size) + "\n"

    # Model
    lm_model = WordLM

    print "Begin testing..."
    # If using gpu..
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    # add parameters to the tf session -> tf.Session(config=gpu_config)
    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = lm_model(args, is_training=False, is_testing=True)

        # save only the last model
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        test_perplexity = run_epoch(sess, mtest, test_data, data_loader, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)
        print("Test time: %.0f" % (time.time() - start))


if __name__ == '__main__':
    main()
