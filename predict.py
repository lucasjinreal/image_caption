"""
this file is to predict single image generate description.
also called Show And Tell

call this script like:

python3 predict.py -p images/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

import argparse
import cv2
import numpy as np


def predict(args_):
    checkpoint_path = args_.checkpoint_path
    words_file = args_.words_file
    image_file = args_.path
    if not os.path.exists(checkpoint_path):
        print('checkpoint path is not exist.')
        exit(0)
    if not os.path.exists(words_file):
        print('words file not found.')
        exit(0)
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpoint_path)
    g.finalize()

    vocab = vocabulary.Vocabulary(words_file)

    if os.path.isdir(image_file):
        with tf.Session(graph=g) as sess:
            restore_fn(sess)
            generator = caption_generator.CaptionGenerator(model, vocab)
            # sent a directory contains images
            file_names = [os.path.join(image_file, i) for i in os.listdir(image_file) if i.lower().endswith('.jpg')
                          or i.lower().endswith('jpeg') or i.lower().endswith('png')]
            file_names = [i for i in file_names if os.path.isfile(i)]
            for f in file_names:
                with tf.gfile.GFile(f, "rb") as img_file:
                    image = img_file.read()
                captions = generator.beam_search(sess, image)
                print("Captions for image %s:" % os.path.basename(f))
                for i, caption in enumerate(captions):
                    # Ignore begin and end words.
                    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                    sentence = " ".join(sentence)
                    print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                # cv2 show image
                image_array = cv2.imread(f, cv2.COLOR_BGR2RGB)
                cv2.imshow('image', image_array)
                cv2.waitKey(0)

    elif os.path.isfile(image_file):
        # sent a single image file
        with tf.Session(graph=g) as sess:
            restore_fn(sess)
            generator = caption_generator.CaptionGenerator(model, vocab)
            # sent a directory contains images
            with tf.gfile.GFile(image_file, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            print("Captions for image %s:" % os.path.basename(f))
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

    else:
        print('image path: {} not found.'.format(image_file))
        exit(0)


def parse_args():
    parser = argparse.ArgumentParser('Show And Tell!')
    parser.add_argument('-c', '--checkpoint_path', default='./model', help='path contains model checkpoint file.')
    parser.add_argument('-w', '--words_file', default='./data/word_counts.txt', help='words_count contains words.')
    parser.add_argument('-p', '--path',  default='./images', help='image or path contains image.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    predict(args_=args)



