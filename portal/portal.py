from typing import Any, List

import os
import sys
import json
import argparse
import subprocess

FILE_NAME = '\\.portalconf'


Config = Any


def create_conf_file(file_name: str) -> Config:
    config: Config = {
        "portals": {}
    }

    with open(file_name, "w") as f:
        json.dump(config, f)

    return config


def write_conf_file(file_name: str, config: Config) -> Config:
    with open(file_name, "w") as f:
        json.dump(config, f)
    return config


def read_conf_file(file_name) -> Config:
    res: Config = {}

    with open(file_name, "r") as f:
        res = json.load(f)

    return res


def create_portal(config: Config, alias: str, path: str) -> Config:
    config['portals'][alias] = {
        'path': path
    }
    return config


def delete_portal(config: Config, alias: str) -> Config:
    del config['portals'][alias]
    return config


def add_r_whitespace(string: str, size: int) -> str:
    if len(string) < size:
        string += " " * (size - len(string))
    return string


def list_portals(config: Config) -> Config:

    max_len = max([len(key) for key in config])

    for key, value in config["portals"].items():
        print(add_r_whitespace(key, max_len + 5), value['path'])

    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates portals")

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        '-l', '--link',
        metavar="PATH",
        help="The path that should be linked to this name")

    group.add_argument(
        '-r', '--remove', action='store_const', const=True,
        help="If set removes positional arguments")

    parser.add_argument(
        '--reset', action='store_const', const=True,
        help="Resets the config")

    parser.add_argument(
        '-old', '--old', action='store_const', const=True,
        help="If set removes positional arguments")

    parser.add_argument(
        '-c', '--code', action='store_const', const=True,
        help="If set removes positional arguments")

    parser.add_argument(
        '--list', action='store_const', const=True,
        help="If set removes positional arguments")

    parser.add_argument(
        'name',
        default="",
        nargs="?",
        help="name of the portal")

    args = parser.parse_args(sys.argv[1:])
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    config_path = script_path + FILE_NAME

    if args.reset or not os.path.exists(config_path):
        create_conf_file(config_path)

    config = read_conf_file(config_path)

    if args.list is not None:
        list_portals(config)
    else:
        if not args.name:
            print("Error: Argument name is required for this action.")
            parser.print_usage()
        elif args.link is not None:
            create_portal(config, args.name, os.path.abspath(args.link))
        elif args.remove is not None:
            delete_portal(config, args.name)
        else:
            if args.name not in config["portals"]:
                print("The portal doesn't exist")
            else:
                portal = config["portals"][args.name]
                CREATE_NO_WINDOW = 0x08000000

                subprocess.call(
                    f"wt -p pws -d \"{portal['path']}\"", creationflags=CREATE_NO_WINDOW)
                if args.code:
                    my_env = os.environ.copy()
                    subprocess.call(f"code \"{portal['path']}\"", shell=True, env=my_env,
                                    creationflags=CREATE_NO_WINDOW)

    write_conf_file(config_path, config)
