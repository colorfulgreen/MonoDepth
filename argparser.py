import argparse


class ArgParser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        p = self.parser
        p.add_argument('--data-path', help='path to load dataset')
        p.add_argument('--log-path', help='path to save logs')

    def args(self):
        return self.parser.parse_args()
