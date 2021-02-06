from typing import List, Any, Set, Optional

import os
import sys
import re
import argparse

Predicate = Any
Caster = Any
TableAction = Any

# --------------- STR MANIPULATION


def lrpad(string: str, size: int, char: str = " "):
    padding = size - len(string)

    if padding <= 0:
        return string

    left_padding = padding // 2
    right_padding = padding - left_padding

    return left_padding * char + string + right_padding * char


# --------------- CASTING


def to_int(value: Any) -> int:
    try:
        res = int(value)
        return res
    except:
        return None


def to_str(value: Any) -> str:
    try:
        res = str(value)
        return res
    except:
        return None


# --------------- DOMAINS


class AttributeDomain:
    def __init__(self, name: str, cast: Caster):
        self.name = name
        self.cast = cast

    def contains(self, val: Any):
        return self.cast(val) is not None

    def __str__(self):
        return self.name


PREMADE_DOMAINS = {
    "int": AttributeDomain("int", to_int),
    "string": AttributeDomain("string", to_str)
}


class Attribute:
    def __init__(self, name: str, domain: AttributeDomain):
        self.name = name
        self.domain = domain

    def __str__(self):
        return f"{self.name}[{self.domain.name}]"


def def_attribute(name: str, preset: str) -> Attribute:
    return Attribute(name, PREMADE_DOMAINS[preset])


def str_to_attribute(string: str) -> Optional[Attribute]:
    match = re.search(r"(\w*)\[(\w*)\]", string)
    if match:
        return Attribute(match.group(1), PREMADE_DOMAINS[match.group(2)])
    else:
        return None


# --------------- TABLE


def line_to_str(line: List[Any], col_width: int):
    res = ""
    for value in line:
        res += lrpad(str(value), col_width) + "|"
    return res[:-1]


def line_to_csv(line: List[Any], delimeter: str = ";"):
    res = ""
    for index, value in enumerate(line):
        res += str(value)
        if index < len(line) - 1:
            res += delimeter
    return res


class Table:
    def __init__(self, name: str,
                 attributes: List[Attribute],
                 key_index: int = 0):
        self.name = name
        self.attributes = attributes
        self.lines = []
        self.key_index = key_index
        self.cols = {}
        for index, attribute in enumerate(attributes):
            self.cols[attribute.name] = index
        self.keys = set()

    def check_line(self, line):
        if len(line) != len(self.attributes):
            return False

        for table_attr, line_value in zip(self.attributes, line):
            if not table_attr.domain.contains(line_value):
                return False

        return True

    def add_line(self, line):

        if line[self.key_index] in self.keys:
            raise "Key already present"

        if self.check_line(line):
            self.lines.append(line)
            self.keys.add(line[self.key_index])
        else:
            raise "Invalid row"

    def remove_line(self, line):
        i = 0
        while i < len(self.lines):
            if self.lines[i][self.key_index] == line[self.key_index]:
                self.lines.pop(i)
                break
            i += 1

    def remove_lines(self, lines):
        for line in lines:
            self.remove_line(line)

    def __getitem__(self, index):
        return self.cols[index]

    def __str__(self):

        col_width = 20
        total_width = len(self.attributes) * col_width + \
            len(self.attributes) - 1
        divider = "\n" + (total_width) * "-" + "\n"

        res = lrpad(self.name, total_width)
        res += "\n\n"
        res += line_to_str(self.attributes, 20)

        res += divider

        for line in self.lines:
            res += line_to_str(line, 20)
            res += divider

        return res


# --------------- TABLE IO


def load_table_from(file_path: str) -> Table:
    with open(file_path, "r", encoding="utf-8") as f:

        [table_name, index] = next(f).strip().split(";")

        header = next(f)
        attributes = [str_to_attribute(x) for x in header.split(";")]

        table = Table(table_name, attributes)
        table.key_index = int(index)

        for index, line in enumerate(f):
            table.add_line(line.strip().split(";"))

        return table


def write_table_to(table: Table, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(table.name + ";" + str(table.key_index) + "\n")

        f.write(line_to_csv(table.attributes) + "\n")

        for index, line in enumerate(table.lines):
            f.write(line_to_csv(line))
            if index < len(table.lines) - 1:
                f.write("\n")


# --------------- TABLE ACTIONS

class TableActionNode:
    def __init__(self,
                 table_action: TableAction,
                 children: List['TableActionNode']):
        self.table_action = table_action
        self.children = children

    def result(self) -> List[Table]:
        children_res = []
        for child in self.children:
            children_res += child.result()
        return self.table_action(children_res)


def load_table_factory(storage_path: str, table_name: str) -> TableAction:
    def load(tables: List[Table]) -> Table:
        new_table = load_table_from(storage_path + "\\" + table_name + ".tbl")
        return [new_table]
    return load


def save_table_factory(storage_path: str = None) -> TableAction:
    def save(tables: List[Table]) -> Table:
        for table in tables:
            file_path = storage_path + "\\" + table.name + ".tbl"
            write_table_to(table, file_path)
        return tables
    return save


def create_table_factory(name: str, attributes: List[Attribute]) -> TableAction:
    def create_table(tables: List[Table]) -> Table:
        new_table = Table(name, attributes)
        return [new_table]
    return create_table


def selection_factory(predicate: Predicate) -> TableAction:
    def selection(tables: List[Table]) -> Table:
        if len(tables) != 1:
            raise (f"Selection is unary, the number of arguments" +
                   f"{len(tables)} doesn't match")

        table = tables[0]

        new_table = Table("selection_table", table.attributes)

        for line in table.lines:
            if predicate(line, table):
                new_table.add_line(line)
        return [new_table]


def projection_factory(attributes: List[str]) -> TableAction:

    def projection(tables: List[Table]) -> Table:

        if len(tables) != 1:
            raise (f"Projection is unary, the number of arguments" +
                   f"{len(tables)} doesn't match")

        table = tables[0]

        new_attributes = []

        for attr in attributes:
            if attr not in table.cols:
                raise "Attribute not in table"
            new_attributes.append(table.attributes[table.cols[attr]])

        new_table = Table("projection_table", new_attributes)

        for line in table.lines:
            new_line = []
            for attr in attributes:
                new_line.append(line[table.cols[attr]])
            new_table.add_line(new_line)
        return [new_table]

    return projection


def rename_factory(name: str) -> TableAction:
    def rename(tables: Table):
        for table in tables:
            table.name = name
        return tables
    return rename


def insert_factory(lines: List[List[Any]]):

    def insert(tables: List[Table]) -> Table:

        if len(tables) != 1:
            raise (f"Projection is unary, the number of arguments" +
                   f"{len(tables)} doesn't match")
        table = tables[0]
        for line in lines:
            table.add_line(line)
        return [table]

    return insert


def const(bl: bool) -> Predicate:
    def predicate(line, table) -> bool:
        return bl
    return predicate


def resolve_stack(section: str, stack: List[str]):
    if section == "select":
        attributes = stack
        return projection_factory(attributes)
    if section == "from":
        name = stack
        return load_table_factory(".", name[0])


def parse_predicate(string: str) -> Predicate:
    return const(True)


def parse_input(string: str) -> TableActionNode:

    string = string.strip()

    blocks = string.split(".")
    storage_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    built_blocks = []

    for block in blocks:
        words_old = block.strip().split()
        words = block.lower().strip().split()

        keyword = words[0].lower()

        if keyword == "create":
            attributes = [str_to_attribute(x) for x in words[2:]]
            built_blocks.append(create_table_factory(words[1], attributes))
        elif keyword == "project":
            built_blocks.append(projection_factory(words[1:]))
        elif keyword == "load":
            built_blocks.append(load_table_factory(storage_path, words[1]))
        elif keyword == "save":
            built_blocks.append(save_table_factory(storage_path))
        elif keyword == "select":
            built_blocks.append(selection_factory(parse_predicate(words[1:])))
        elif keyword == "insert":
            args = [x.strip() for x in " ".join(words_old[1:]).split(",")]
            built_blocks.append(insert_factory([args]))
        elif keyword == "rename":
            built_blocks.append(rename_factory(words[1]))
    return built_blocks


def build_tree(blocks: List[TableAction]):

    root = None
    last = None

    for block in blocks:
        current = TableActionNode(block, [])
        if last:
            last.children.append(current)

        if not root:
            root = current

        last = current
    return root


def get_args() -> None:
    parser = argparse.ArgumentParser(
        description="Execute commands on database")

    parser.add_argument('query', type=str, help="the main input")

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    args = get_args()
    print(build_tree(parse_input(args.query)).result()[0])
