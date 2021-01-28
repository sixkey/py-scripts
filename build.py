import PyInstaller.__main__

if __name__ == "__main__":

    names = ["viminimg"]

    for name in names:
        PyInstaller.__main__.run([
            f"./{name}/{name}.py",
            "--onefile",
        ])
