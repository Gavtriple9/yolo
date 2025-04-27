import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='prog')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode', default=False)
    parser.add_argument('-l', '--log_level', type=str, help='set log level', default="INFO")
    
    parser_train = subparsers.add_parser('train', help='train the model')
    parser_test = subparsers.add_parser('test', help='train the model')

    return parser