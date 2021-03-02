import PyInstaller.__main__
import sys
import os
import json


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds py-scripts scripts")

    parser.add_argument(
        'keys', nargs="+",
        help="keys")

    args = parser.parse_args(sys.argv[1:])

    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    config_path = os.path.join(script_path, ".buildconfig")

    profiles = None
    with open(config_path, "r") as f:
        profiles = json.load(f)["profiles"]

    if not profiles:
        print("Builder has no config")
        sys.exit(0)

    keys = args.keys

    if "*" in keys:
        keys = profiles.keys()

    print("Building: " + str(keys))

    for key in keys:
        if key not in profiles:
            print(f"Error: key {key} doesn't exist")
            continue

        profile = profiles[key]

        for build in profile:
            path = build["path"]
            flags = build["flags"] if "flags" in build else []
            newname = build["newname"] if "newname" in build else None

            if newname:
                flags.append(f"-n{newname}")

            PyInstaller.__main__.run([
                path,
                "--onefile",
                *flags
            ])
