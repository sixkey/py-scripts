import PyInstaller.__main__
import sys

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds py-scripts scripts")

    parser.add_argument(
        'keys', nargs="+",
        help="keys")

    args = parser.parse_args(sys.argv[1:])

    paths = {
        "viminimg": [{"path": "viminimg/viminimg.py"}],
        "portal": [{"path": "portal/portal.py"}, {"path": "portal/portal.py", "newname": "port", "flags": ["--noconsole"]}]
    }

    keys = args.keys

    if not keys:
        keys = paths.keys()

    print("Building: " + str(keys))

    for key in keys:
        if key not in paths:
            print(f"Error: key {key} doesn't exist")
            continue

        profile = paths[key]

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
