"""
This file contains the arguments to parse at command line.
File main.py will call get_args, which then the arguments will be returned
"""
import argparse


def get_args():
    """
    Parses arguments at command line
    :return: args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train')  # mode can be train or test
    parser.add_argument('--model', dest='model', type=str, default='')  # model filename
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v1')  # critic model filename

    args = parser.parse_args()

    return args
