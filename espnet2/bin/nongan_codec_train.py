#!/usr/bin/env python3
from espnet2.tasks.nongan_codec import NonGANCodecTask


def get_parser():
    parser = NonGANCodecTask.get_parser()
    return parser


def main(cmd=None):
    """GAN-based Codec training

    Example:

        % python gan_codec_train.py --print_config --optim1 adadelta
        % python gan_codec_train.py --config conf/train.yaml
    """
    NonGANCodecTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

