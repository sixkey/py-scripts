from typing import Callable, List, Set, Tuple, Any, Optional, Dict

import os
import sys
import glob
from shutil import rmtree, copyfile
import json
import argparse

from PIL import Image


def log(*args, **kwargs) -> None:
    print(*args, **kwargs)


def log_mute(*args, **kwargs) -> None:
    pass


def transform_path(file: str) -> str:
    return NEW_PATH + file[len(ORG_PATH):]


def transform_path_filename(path: str,
                            transformer: Callable[[
                                str, int], str] = lambda x, i: x,
                            index: int = 0) -> str:
    filename = os.path.basename(path)
    name, extension = os.path.splitext(path)
    name = name.split("\\")[-1]
    return path[:-len(filename)] + transformer(name, index) + extension


def change_extension(path: str, new_extension: str) -> str:
    name, extension = os.path.splitext(path)
    return name + "." + new_extension


# IMAGE EDITING

def create_open_file():
    def open_file(original_picture, picture, file_name, original_file_name, index):
        picture = Image.open(file_name)
        original_picture = picture
        return original_picture, picture, file_name, original_file_name, index
    return open_file


def create_rollback():
    def rollback(original_picture, picture, file_name, original_file_name, index):
        return original_picture, original_picture, original_file_name, original_file_name, index
    return rollback


def create_rename(transformer: Callable[[str, int], str] = lambda x, y: x):
    def create(original_picture, picture, file_name, original_file_name, index):
        return original_picture, picture, transformer(file_name, index), original_file_name, index
    return create


def create_rename_from_org(transformer: Callable[[str, int], str] = lambda x, y: x):
    def create(original_picture, picture, file_name, original_file_name, index):
        return original_picture, picture, transformer(original_file_name, index), original_file_name, index
    return create


def create_rescale(max_size: int, height: bool = False):
    def rescale(original_picture, picture, file_name, original_file_name, index):
        ratio = max_size / \
            picture.size[0] if not height else max_size / picture.size[1]
        picture = picture.resize((int(picture.size[0] * ratio),
                                  int(picture.size[1] * ratio)),
                                 Image.ANTIALIAS)
        return original_picture, picture, file_name, original_file_name, index
    return rescale


def create_save(quality: Optional[int] = None):
    def save(original_picture, picture, file_name, original_file_name, index):
        if quality is None:
            picture.save(file_name)
        else:
            picture.save(file_name, optimize=True, quality=quality)
        return original_picture, picture, file_name, original_file_name, index
    return save


def create_convert():
    def convert(original_picture, picture, file_name, original_file_name, index):
        picture = picture.convert('RGB')
        return original_picture, picture, file_name, original_file_name, index
    return convert


def build_transformer(bricks):
    def transformer(original_picture, picture, file_name, original_file_name, index):
        for brick in bricks:
            result = brick(
                original_picture, picture, file_name, original_file_name, index)
            original_picture, picture, file_name, original_file_name, index = result
        return original_picture, picture, file_name, original_file_name, index
    return transformer


def build_file_transformer(bricks, res_image, res_files):
    transformer = build_transformer(bricks)

    def file_transformer(original_picture, picture, file_name, original_file_name, index):
        transformer(original_picture, picture,
                    file_name, original_file_name, index)
        return res_image, res_files
    return file_transformer


def remove_folder(folder: str) -> None:
    if os.path.exists(folder):
        rmtree(folder)


def create_folder(folder: str) -> None:
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print("Folder already exists")


def add_tuples(xs: Tuple[Any, ...], ys: Tuple[Any, ...]) -> Tuple[Any, ...]:
    res = []
    for x, y in zip(xs, ys):
        res.append(x + y)
    return tuple(res)


FolderMetadata = Tuple[str, Tuple[int, int], Tuple[int, int], List[str]]


def compress_folder(folder: str,
                    res_container: List[FolderMetadata],
                    file_transformer):
    create_folder(transform_path(folder))

    total = 0, 0
    total_here = 0, 0

    image_files = glob.glob(folder + "*.png")
    image_files.extend(glob.glob(folder + "*.jpg"))

    for index, file in enumerate(image_files):
        file_count = file_transformer(None, None, file, file, index)
        total_here = add_tuples(total_here, file_count)
    total = total_here

    for index, file in enumerate(glob.glob(folder + '.kgal')):
        copyfile(file, transform_path(file))

    children = []

    for fldr in glob.glob(folder + "*\\"):
        folder_file_count = compress_folder(
            fldr, res_container, file_transformer)
        children.append(os.path.normpath(fldr))
        total = add_tuples(total, folder_file_count)

    res_container.append(
        (os.path.normpath(folder), total_here, total, children))

    return total


class Node:
    def __init__(self, name: str, data: FolderMetadata, children: List['Node'],
                 parent: Optional['Node']):
        self.name = name
        self.children = children
        self.parent = parent
        self.data = data


def folder_annotation(metadata: List[FolderMetadata]) -> Any:
    if metadata == []:
        return {}

    nodes: Dict[str, Node] = {}

    for record in metadata:
        nodes[record[0]] = Node(record[0], record, [], None)

    for folder, _, _, children in metadata:
        for child in children:
            nodes[child].parent = nodes[folder]
            nodes[folder].children.append(nodes[child])

    root = None
    for key in nodes:
        if nodes[key].parent == None:
            root = nodes[key]
            break

    return annotate_node(root)


def annotate_node(current: Node) -> Optional[Dict[str, Any]]:

    path, here, total, _ = current.data

    tags = path.split("\\")

    res = {
        'children': [],
        'path': path,
        'totalImages': total[0],
        'totalFiles': total[1],
        'hereImages': here[0],
        'hereFiles': here[1],
        'tags': tags,
        'keytag': tags[-1] if tags else ''
    }

    if os.path.exists(path + "\\" + '.vminimeta'):
        with open(path + "\\" + '.vminimeta', 'r', encoding="utf-8") as f:
            meta = json.load(f)
            if 'tags' in meta:
                res['tags'] += meta['tags']
            if 'keytag' in meta:
                res['keytag'] = meta['keytag']

    for child in current.children:
        res['children'].append(annotate_node(child))

    if not res['children']:
        del res['children']

    return res


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-i', '--indexed', action='store_const', const=True)
    parser.add_argument('--resfolder')
    parser.add_argument('--verbose', action='store_const', const=True)
    parser.add_argument('orgfolder', type=dir_path)

    args = parser.parse_args(sys.argv[1:])

    ORG_PATH = args.orgfolder
    if ORG_PATH[-1] == '\\' or ORG_PATH[-1] == '//':
        ORG_PATH = ORG_PATH[:-1]
    NEW_PATH = args.resfolder if args.resfolder else ORG_PATH + "-viminimg"

    metadata_path = NEW_PATH + "-metadata.json"
    ORG_PATH += '\\'
    NEW_PATH += "\\"

    remove_folder(NEW_PATH)

    if not args.verbose:
        log = log_mute

    transformer = build_file_transformer([
        create_open_file(),

        create_rename_from_org(lambda x, i: transform_path(x)),
        create_rename(lambda x, i: transform_path_filename(
            x, lambda x, i: str(i), i)),
        create_save(),

        create_rename_from_org(lambda x, i: transform_path(x)),
        create_rename(lambda x, i: transform_path_filename(
            x, lambda x, i: str(i) + "-s", i)),
        create_rescale(600),
        create_save(),

        create_rename_from_org(lambda x, i: transform_path(x)),
        create_rename(lambda x, i: transform_path_filename(
            x, lambda x, i: str(i) + "-c", i)),
        create_save(10),

    ], 1, 1)

    metadata_container = []
    compress_folder(ORG_PATH, metadata_container, transformer)
    annotation = folder_annotation(metadata_container)

    metadata = {
        'folders': metadata_container,
        'tree': annotation
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
