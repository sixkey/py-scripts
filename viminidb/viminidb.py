from typing import List, Any, Set, Optional, Tuple, Callable

import os
import sys
import re
import argparse
from collections import deque

Predicate = Any
Caster = Any
TableAction = Any

# --------------- STR MANIPULATION


def concat(strings: List[str]) -> str:
    res = ""
    for string in strings:
        res += string
    return res


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


def string_to_bool(value: str) -> bool:
    return value.strip() == "True"


def to_bool(value: Any) -> bool:
    try:
        if isinstance(value, str):
            res = string_to_bool(value)
        else:
            res = bool(value)
        return res
    except:
        return None


def to_float(value: Any) -> float:
    try:
        res = float(value)
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
    "str": AttributeDomain("str", to_str),
    "bool": AttributeDomain("bool", to_bool),
    "float": AttributeDomain("float", to_float)
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

        res_line = []
        if len(line) != len(self.attributes):
            return None

        for table_attr, line_value in zip(self.attributes, line):
            if not table_attr.domain.contains(line_value):
                return None
            else:
                res_line.append(table_attr.domain.cast(line_value))
        return res_line

    def add_line(self, line):
        if line[self.key_index] in self.keys:
            raise "Key already present"
        casted_line = self.check_line(line)
        if casted_line is not None:
            self.lines.append(casted_line)
            self.keys.add(casted_line[self.key_index])
        else:
            raise RuntimeError(f"Invalid row: {line} for {self.name}")

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

        res = "\n" + lrpad(self.name, total_width)
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
            line_values = line.strip().split(";")
            table.add_line(line_values)

        return table


def write_table_to(table: Table, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(table.name + ";" + str(table.key_index) + "\n")

        f.write(line_to_csv(table.attributes) + "\n")

        for index, line in enumerate(table.lines):
            f.write(line_to_csv(line))
            if index < len(table.lines) - 1:
                f.write("\n")


def list_tables(storage_path: str) -> List[str]:

    res = []

    for _, _, files in os.walk(storage_path, topdown=False):
        res += [os.path.splitext(f)[0] for f in files]

    return res

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


def get_local_storage() -> str:
    return os.path.dirname(os.path.realpath(sys.argv[0])) + "\\.viminidbstorage"


def load_table_factory(storage_path: str, table_name: str) -> TableAction:
    def load(table: Table) -> Table:
        new_table = load_table_from(storage_path + "\\" + table_name + ".tbl")
        return new_table
    return load


def save_table_factory(storage_path: str, table_name: str = None) -> TableAction:
    def save(table: Table) -> Table:
        if table:
            file_name = (table_name if table_name else table.name) + ".tbl"
            file_path = storage_path + "\\" + file_name
            write_table_to(table, file_path)
            return table
    return save


def create_table_factory(name: str, attributes: List[Attribute]) -> TableAction:
    def create_table(table: Table) -> Table:
        new_table = Table(name, attributes)
        return new_table
    return create_table


def selection_factory(predicate: Predicate) -> TableAction:
    def selection(table: Table) -> Table:
        new_table = Table("selection_table", table.attributes)

        for line in table.lines:
            if predicate(line, table):
                new_table.add_line(line)
        return new_table
    return selection


Row = Tuple[Any, ...]


def projection_factory(row_functions: List[Callable[[Row], Any]], attributes: List[Attribute]) -> TableAction:

    def projection(table: Table) -> Table:
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
        return new_table

    return projection


def rename_factory(name: str) -> TableAction:
    def rename(table: Table):
        table.name = name
        return table
    return rename


def insert_factory(function):

    def insert(table: Table) -> Table:
        lines = function(table)
        for line in lines:
            table.add_line(line)
        return table

    return insert


def join_factory(type: str, left: TableAction, right: TableAction, condition):
    def join(table: Table) -> Table:
        left_table = left(table)
        right_table = right(table)

        final_table = None

        if not condition:
            same_attributes = []
            final_attributes = left_table.attributes[:]
            right_indexes = []
            for r_attribute in right_table.attributes:
                if r_attribute.name in left_table.cols:
                    same_attributes.append(r_attribute)
                else:
                    right_indexes.append(right_table.cols[r_attribute.name])
                    final_attributes.append(r_attribute)

            def final_transformer(left, right):
                return left + [right[index] for index in right_indexes]
            final_table = Table("joined_table", final_attributes)

            for l_line in left_table.lines:
                for r_line in right_table.lines:
                    add = True
                    for attr in same_attributes:
                        if l_line[left_table.cols[attr.name]] != r_line[right_table.cols[attr.name]]:
                            add = False
                            break
                    if add:
                        final_table.add_line(final_transformer(l_line, r_line))

            return final_table
    return join


def print_factory():
    def print_tbl(table: Table):
        print(table)
        return table
    return print_tbl


def nothing_factory() -> TableAction:
    def nothing(table: Table) -> Table:
        return table
    return nothing


def dot(a: TableAction, b: TableAction):
    def dot_joined(table: Table) -> Table:
        return a(b(table))
    return dot_joined


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


# --------------- QUERY PARSING


def make_tuple(x, y):
    if isinstance(y, tuple):
        try:
            return (x, *y)
        except:
            return (x, y)
    else:
        return (x, y)


BINARY_OPERATIONS = {
    ".": lambda x, y: dot(x, y),
    ",": lambda x, y: make_tuple(x, y),
    "&&": lambda x, y: x and y,
    "||": lambda x, y: x or y,

    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "<=": lambda x, y: x <= y,
    ">=": lambda x, y: x >= y,
    "<": lambda x, y: x < y,
    ">": lambda x, y: x > y,

    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "%": lambda x, y: x % y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "^": lambda x, y: x ** y
}


OPERATORS_LEVELS = {
    ".": 1,
    ",": 2,

    "&&": 3,
    "||": 3,

    "==": 4,
    "!=": 4,
    "<=": 4,
    ">=": 4,
    "<": 4,
    ">": 4,

    "+": 5,
    "-": 5,
    "%": 6,
    "*": 6,
    "/": 6,
    "^": 7
}

DB_FUNCTIONS_ARITIES = {
    "load": 1,
    "save": (0, 1),
    "project": (1, "*"),
    "create": (2, "*"),
    "insert": 1,
    "rename": (1, "*"),
    "print": 0,
    "select": 1,
    "join": (3, 4),
    "nothing": 0,
}

DB_FUNCTIONS = [x for x in DB_FUNCTIONS_ARITIES]
OPERATORS = [x for x in OPERATORS_LEVELS]

Token = Tuple[str, str]


class Variable:
    def __init__(self, var_type, name):
        self.var_type = var_type
        self.name = name

    def get(self):
        return ("var", self.var_type, self.name)

    def __str__(self):
        return (f"{self.var_type} {self.name}")


class Constant:
    def __init__(self, val_type, value):
        self.val_type = val_type
        self.value = value

    def get(self):
        return ("val", self.val_type, self.value)

    def __str__(self):
        return str(self.value) + " :: " + self.val_type


VAL_TESTERS = {
    "int": lambda x: isinstance(x, int),
    "float": lambda x: isinstance(x, float),
    "tuple": lambda x: isinstance(x, tuple),
    "bool": lambda x: isinstance(x, bool),
    "fun": callable,
    "str": lambda x: isinstance(x, str),
}


def result_to_val_tuple(result: Any) -> Tuple[str, str, Any]:
    val_type = "any"
    for name, tester in VAL_TESTERS.items():
        if tester(result):
            val_type = name
            break
    return "val", val_type, result


def build_row_expression(left, right, operation) -> Tuple[str, str, Any]:
    left_state, left_type, left_val = left
    right_state, right_type, right_val = right

    if left_state == "val" and right_state == "val":
        if left_type != "row_function" and right_type != "row_function":
            return result_to_val_tuple(operation(left_val, right_val))

    def expression(line, table) -> bool:
        left_final_value = None
        if left_state == "var":
            left_final_value = line[table.cols[left_val]]
        elif left_state == "val":
            if left_type == "row_function":
                left_final_value = left_val(line, table)
            else:
                left_final_value = left_val

        right_final_value = None

        if right_state == "var":
            right_final_value = line[table.cols[right_val]]
        elif right_state == "val":
            if right_type == "row_function":
                right_final_value = right_val(line, table)
            else:
                right_final_value = right_val

        return operation(left_final_value, right_final_value)

    return "val", "row_function", expression


def tokenize_word(string: str) -> Token:

    if string in DB_FUNCTIONS:
        return ("function", string)
    if string in ["(", ")"]:
        return ("bracket", string)
    if string in OPERATORS:
        return ("operator", string)
    else:

        word_type = "variable"

        for _, (pattern, _) in VALUE_PARSINGS.items():
            if re.match(pattern, string):
                word_type = "value"
                break

        return (word_type, string)


def tokenize_input(string: str) -> List[Token]:

    tokens = []

    word_buffer = ''
    quote_started = False

    for c in string + " ":

        if c == "'":
            if word_buffer and not quote_started:
                raise ("Error occured whilte parsing text, check quotations" +
                       " marks")

            if quote_started:
                tokens.append(("value", word_buffer))
                word_buffer = ""
            quote_started = not quote_started

        elif c in [" ", "(", ")"]:
            if quote_started:
                raise ("Quotes not ended properly, check quotation marks")
            if word_buffer:
                tokens.append(tokenize_word(word_buffer))
                word_buffer = ""
            if c in ["(", ")"]:
                tokens.append(tokenize_word(c))
        else:
            word_buffer += c

    return tokens


def check_args(function_name: str, args) -> bool:
    # CHECK ARITY
    arity = DB_FUNCTIONS_ARITIES[function_name]

    args_num = len(args)
    if isinstance(arity, int):
        return args_num == arity
    elif isinstance(arity, tuple):
        lower_bound, upper_bound = arity
        res = True
        if isinstance(lower_bound, int):
            res &= args_num >= lower_bound
        if isinstance(upper_bound, int):
            res &= args_num <= upper_bound

        return res

    return False


def resolve(element):
    return element


VALUE_PARSINGS = {
    "int": (r"\d+", int),
    "float": (r"\d+.\d+", float),
    "bool": (r"True|False", string_to_bool),
}


def parse_miniblock(tokens: List[Token]) -> TableActionNode:
    storage_path = get_local_storage()
    head_type, head_val = tokens[0]
    if len(tokens) == 1:
        if head_type == 'variable':
            return Variable("any", head_val)
        if head_type == 'value':
            constant_type = "str"
            constant_val = head_val

            for var_type, (pattern, caster) in VALUE_PARSINGS.items():
                if re.match(pattern, head_val):
                    constant_val = caster(head_val)
                    constant_type = var_type
                    break

            return Constant(constant_type, constant_val)
        if head_type == 'parsed':
            return head_val

    if head_type == 'function':
        arg_tokens = [resolve(parse_miniblock([t])) for t in tokens[1:]]
        function = None

        if not check_args(head_val, arg_tokens):
            raise RuntimeError(f"Invalid arguments near {head_val}")
        arg_num = len(arg_tokens)

        if head_val == "project":
            function = projection_factory([x.name for x in arg_tokens])

        if head_val == "load":
            function = load_table_factory(storage_path, arg_tokens[0].name)

        elif head_val == "save":
            table_name = None
            if len(arg_tokens) == 1:
                table_name = arg_tokens[0].name
            function = save_table_factory(storage_path, table_name)

        elif head_val == "create":
            attributes = [str_to_attribute(x.name) for x in arg_tokens[1:]]
            function = create_table_factory(arg_tokens[0].name, attributes)

        elif head_val == "insert":

            print(arg_tokens[0].val_type)
            if arg_tokens[0].val_type == "tuple":
                function = insert_factory(lambda x: [arg_tokens[0].value])
            elif arg_tokens[0].val_type == "fun":
                function = insert_factory(
                    lambda x: arg_tokens[0].value(x).lines)
            else:
                function = insert_factory(lambda x: [[arg_tokens[0].value]])
        elif head_val == "select":
            function = selection_factory(arg_tokens[0].get()[2])

        elif head_val == "print":
            function = print_factory()

        elif head_val == "nothing":
            function = nothing_factory()

        elif head_val == "rename":
            function = rename_factory(arg_tokens[0].name)

        elif head_val == "join":
            left = arg_tokens[1 if arg_num == 3 else 2].get()[2]
            right = arg_tokens[2 if arg_num == 3 else 3].get()[2]
            condition = None if arg_num == 3 else arg_tokens[1].get()[2]
            function = join_factory(
                arg_tokens[0].name, left, right, condition)
        return Constant("table_transform", function)


def parse_tokens(tokens: List[Token]) -> TableActionNode:

    brackets = []
    bracket_stack = []

    for index, (token_type, token_value) in enumerate(tokens):
        if token_type == "bracket":
            if token_value == "(":
                bracket_stack.append(index)
            else:
                if not bracket_stack:
                    raise "Missing starting bracket"
                start = bracket_stack.pop()

                if not bracket_stack:
                    brackets.append((start, index))

    offset = 0
    for (left, right) in brackets:
        tokens = tokens[:left - offset] + \
            [("parsed", parse_tokens(tokens[left + 1 - offset: right - offset]))
             ] + tokens[right + 1 - offset:]
        offset += right - left

    for level in range(0, 10):
        for index, (token_type, token_value) in enumerate(tokens):
            if token_type == "operator":
                if OPERATORS_LEVELS[token_value] == level:

                    left = parse_tokens(tokens[:index]).get()
                    right = parse_tokens(tokens[index + 1:]).get()

                    e_state, e_type, e_val = build_row_expression(
                        left, right, BINARY_OPERATIONS[token_value])

                    return Constant(e_type, e_val)

    return parse_miniblock(tokens)


def get_args() -> None:
    parser = argparse.ArgumentParser(
        description="Execute commands on database")

    parser.add_argument('-ls', '--list', action="store_const", const=True,
                        help="If set lists present tables")

    parser.add_argument('-s', '--session', action="store_const", const=True,
                        help="If set starts session")

    parser.add_argument('query', nargs="?", type=str, help="the main input")

    return parser.parse_args(sys.argv[1:])


def execute_query(query: str):
    tokens = tokenize_input(query)
    tree = parse_tokens(tokens)

    if isinstance(tree, Constant) and tree.val_type in ("table_transform", "fun"):
        function = tree.get()[2]
        if os.path.exists(storage_path + "\\current.tbl"):
            load_current = load_table_factory(storage_path, "current")
            function = dot(function, load_current)
        save_current = save_table_factory(storage_path, "current")

        final_query = dot(save_current, function)
        print(final_query(None))
    else:
        print(tree)


if __name__ == "__main__":
    args = get_args()

    storage_path = get_local_storage()

    if not os.path.exists(storage_path):
        os.mkdir(storage_path)

    if args.list:
        print(list_tables(storage_path))

    query_que = deque()
    if args.query:
        query_que.append(args.query)
    else:
        user_input = input(">>> ")
        if user_input:
            query_que.append(user_input)
    while query_que:
        query = query_que.popleft()

        execute_query(query)

        if args.session:
            user_input = input(">>> ")
            if user_input:
                query_que.append(user_input)
