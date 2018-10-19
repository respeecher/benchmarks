# General purpose CPU/GPU burner for benchmarking.

import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--framework", type=str, required=True,
    choices=("tf", "pytorch"),
    help="DL framework to use."
)
parser.add_argument(
    "--task", type=str, required=True,
    choices=("matmul", "conv2d", "rnn"),
    help="Task."
)
parser.add_argument(
    "--iterations", type=int, default=100000,
    help="Number of iterations to perform."
)
parser.add_argument(
    "--batch-size", type=int, default=32,
    help="Minibatch size."
)
parser.add_argument(
    "--matrix-size", type=int, default=64,
    help="Matrix size for `matmul` task."
)
parser.add_argument(
    "--depth", type=int, default=64,
    help="Number of convolutional filters for `conv2d` task."
)
parser.add_argument(
    "--kernel-size", type=int, default=3,
    help="Convolutional kernel size for `conv2d` task."
)
parser.add_argument(
    "--input-size-2d", type=int, default=100,
    help="Input size for `conv2d` task."
)
parser.add_argument(
    "--state-size", type=int, default=128,
    help="RNN state size for `rnn` task."
)
parser.add_argument(
    "--sequence-length", type=int, default=30,
    help="Sequence length used in `rnn` task."
)
parser.add_argument(
    "--input-size", type=int, default=128,
    help="RNN input size for `rnn` task."
)

args = parser.parse_args()


def formatter(s, style):
    return "{style}{s}\033[00m".format(style=style, s=s)

def red(s):
    return formatter(s, style="\033[31m")

def green(s):
    return formatter(s, style="\033[32m")

def bold(s):
    return formatter(s, style="\033[01m")

try:
    from tqdm import tqdm
except ImportError:
    print()
    print(
        red("Error:"),
        bold("tqdm"), "progressbar is required for running this script.",
        "Please install it by running `pip install tqdm`\n"
    )
    sys.exit(0)

if args.framework == "tf":
    try:
        import tensorflow as tf
    except ImportError:
        print()
        print(
            red("Error:"),
            "You specified", bold("Tensorflow"), "as a DL framework, but it is not installed.",
            "Please visit https://www.tensorflow.org for installation details.\n"
        )
        sys.exit(0)

elif args.framework == "pytorch":
    try:
        import torch
    except ImportError:
        print()
        print(
            red("Error:"),
            "You specified", bold("PyTorch"), "as a DL framework, but it is not installed.",
            "Please visit https://pytorch.org for installation details.\n"
        )
        sys.exit(0)

if args.framework == "tf":

    print()
    print("Running", bold(args.task), "task in", bold("TensorFlow"))
    print()

    if args.task == "matmul":
        a = tf.random_normal(shape=(args.batch_size, args.matrix_size))
        b = tf.random_normal(shape=(args.matrix_size, args.matrix_size))

        c = tf.matmul(a, b)

        with tf.Session() as sess:
            for _ in tqdm(range(args.iterations)):
                c.eval()

    elif args.task == "conv2d":
        input = tf.random_normal(shape=(args.batch_size, args.input_size_2d, args.input_size_2d, 1))
        filter = tf.random_normal(shape=(args.kernel_size, args.kernel_size, 1, args.depth))

        conv = tf.nn.conv2d(input, filter, (1, 1, 1, 1), padding="SAME")

        with tf.Session() as sess:
            for _ in tqdm(range(args.iterations)):
                conv.eval()

    elif args.task == "rnn":
        input = tf.random_normal(shape=(args.batch_size, args.sequence_length, args.input_size))
        cell = tf.nn.rnn_cell.BasicRNNCell(args.state_size)

        output, state = tf.nn.dynamic_rnn(
            cell, input, sequence_length=args.batch_size * [args.sequence_length],
            dtype=tf.float32
        )

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for _ in tqdm(range(args.iterations)):
                output.eval()

elif args.framework == "pytorch":

    print()
    print("Running", bold(args.task), "task in", bold("PyTorch"))
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.task == "matmul":
        for _ in tqdm(range(args.iterations)):
            a = torch.zeros(args.batch_size, args.matrix_size).normal_().to(device)
            b = torch.zeros(args.matrix_size, args.matrix_size).normal_().to(device)

            torch.mm(a, b)

    elif args.task == "conv2d":

        conv2d = torch.nn.Conv2d(1, args.depth, args.kernel_size).to(device)

        for _ in tqdm(range(args.iterations)):
            input = torch.zeros(args.batch_size, 1, args.input_size_2d, args.input_size_2d).normal_().to(device)

            conv2d(input)

    elif args.task == "rnn":

        rnn = torch.nn.RNN(args.input_size, args.state_size, batch_first=True).to(device)

        for _ in tqdm(range(args.iterations)):
            input = torch.zeros(args.batch_size, args.sequence_length, args.input_size).normal_().to(device)

            rnn(input)