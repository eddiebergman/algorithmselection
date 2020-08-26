import os
import json
import argparse

def valid_file(filepath):
    """
    'Type' for argparse - checks that file exists but does not open it.
    """
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f'{filepath} does not exist')
    return filepath

def info_cmd(config_path):
    """
    Prints out the current state of a configuration based on its <config> and
    what it can find in the containing folder and its progress.json

    Params
    ======
    config_path | filepath
         The config to get info on
    """
    pass

def verify_cmd(config_path):
    """
    Verifies a <config> file and any current state that it may be in based on
    what it can find in its progress.json

    Params
    ======
    config_path | filepath
        The config to verify
    """
    pass

def run_cmd(config_path, step=False):
    """
    Runs the experiment on the <config> with any passed <options>

    Params
    ======
    config_path | filepath
        The config to get info on

    step | bool -> False
        Whether to take a single step or run the whole config
    """
    pass


def create_parser():
    """
    Creates the parser with the commands
    --info
    --verify
    --run
    --step
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--info',
                        type=valid_file,
                        metavar='config_path')

    parser.add_argument('--verify',
                        type=valid_file,
                        metavar='config_path')

    parser.add_argument('-r', '--run',
                        type=valid_file,
                        metavar='config_path')

    parser.add_argument('--step',
                        action='store_true')

    return parser


def pass_args(args):
    """
    Passes and verifies args
    """
    # The filepaths are already verified
    if args.info:
        info_cmd(args.info)

    elif args.verify:
        verify_cmd(args.verify)

    elif args.run:
        run_cmd(args.run, step=args.step)

    else:
        parser.print_help()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    pass_args(args)

