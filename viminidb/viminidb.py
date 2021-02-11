from typing import (List, Any, Set,
                    Optional, Tuple, Callable,
                    Dict, Deque, DefaultDict, Union, Literal)

from videbugging import create_loud_function

import os
import sys
import re
import argparse
from datetime import datetime, date, time
from collections import deque, defaultdict

DateTime = Any
Date = Any
Time = Any
Args = Any
Row = Tuple[Any, ...]
TableFunction = Callable[['Table'], List[Row]]
TableAction = Callable[['Table'], 'Table']

Predicate = Any
Caster = Any

ColDefinition = Tuple['Attribute', bool, Optional[str]]


ENCLOSINGS = {
    "parentheses": ("(", ")"),
    "braces": ('{', "}"),
    "brackets": ("[", "]")
}


RE_PATERNS = {
    "attr_shrt": r"(\w+)\[(\w+)]",
    "attr_lng": r"(\w+)\[(\w+)]\[([\s\S]*)\]",
    "datetime": r"^\d{4}\-\d{2}\-\d{2} \d{2}:\d{2}$",
    "date": r"^\d{4}\-\d{2}\-\d{2}$"
}


FORMATS = {
    'datetime': "%Y-%m-%d %H:%M",
    'date': "%Y-%m-%d"
}


# --------------- MISC


def soft_assert(bl: bool, message: str):
    if not bl:
        raise RuntimeError(message)
    return bl


# --------------- STR MANIPULATION


def starts_with(string: str, substring: str, start: int = 0) -> bool:
    if len(string) - start < len(substring):
        return False

    for i in range(len(substring)):
        if string[start + i] != substring[i]:
            return False
    return True


def concat(strings: List[str]) -> str:
    res = ""
    for string in strings:
        res += string
    return res


def lrpad(string: str, size: int, char: str = " ") -> str:
    padding = size - len(string)

    if padding <= 0:
        return string

    left_padding = padding // 2
    right_padding = padding - left_padding

    return left_padding * char + string + right_padding * char


# --------------- CASTING


def to_int(value: Any) -> Optional[int]:
    try:
        res = int(value)
        return res
    except:
        return None


def to_str(value: Any) -> Optional[str]:
    try:
        res = str(value)
        return res
    except:
        return None


def string_to_bool(value: str) -> bool:
    return value.strip() == "True"


def to_bool(value: Any) -> Optional[bool]:
    try:
        if isinstance(value, str):
            res = string_to_bool(value)
        else:
            res = bool(value)
        return res
    except:
        return None


def to_float(value: Any) -> Optional[float]:
    try:
        res = float(value)
        return res
    except:
        return None


def to_datetime(value: Any) -> DateTime:
    if isinstance(value, str):
        return datetime.strptime(value, FORMATS['datetime'])
    elif isinstance(value, datetime):
        return value
    elif isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    else:
        raise RuntimeError("Invalid format, to_datetime works only for str, " +
                           "date and datetime, your type was: " +
                           str(type(value)) + " and value " + str(value))


def to_date(value: Any) -> Date:
    if isinstance(value, str):
        return datetime.strptime(value, FORMATS['date']).date()
    elif isinstance(value, datetime):
        return value.date
    elif isinstance(value, date):
        return value
    else:
        raise RuntimeError("Invalid format, to_date works only for str, " +
                           "date and datetime, your type was: " +
                           str(type(value)) + " and value " + str(value))


def to_attribute(value: Any) -> 'Attribute':
    if isinstance(value, str):
        attr_name, attr_type = None, None

        match = re.match(RE_PATERNS["attr_shrt"], value)
        if match:
            attr_name, attr_type = match.groups()

        assert attr_name and attr_type, "Doesn't match the attribute pattern"

        return Attribute(attr_name, PREMADE_DOMAINS[attr_type])
    if isinstance(value, Attribute):
        return value
    else:
        raise RuntimeError(
            "Invalid format, to_attribute works only with string")
# --------------- DOMAINS


class AttributeDomain:
    def __init__(self, name: str, cast: Caster):
        self.name = name
        self.cast = cast

    def contains(self, val: Any) -> bool:
        try:
            casted = self.cast(val)
            return casted is not None
        except:
            return False

    def __str__(self) -> str:
        return self.name


class Attribute:
    def __init__(self, name: str, domain: AttributeDomain):
        self.name = name
        self.domain = domain

    def __str__(self) -> str:
        return f"{self.name}[{self.domain.name}]"


PREMADE_DOMAINS = {
    "int": AttributeDomain("int", to_int),
    "float": AttributeDomain("float", to_float),
    "bool": AttributeDomain("bool", to_bool),
    "str": AttributeDomain("str", to_str),
    "date": AttributeDomain("date", to_date),
    "datetime": AttributeDomain("datetime", to_datetime)
}


VAL_TESTERS = {
    "int": lambda x: isinstance(x, int),
    "float": lambda x: isinstance(x, float),
    "tuple": lambda x: isinstance(x, tuple),
    "bool": lambda x: isinstance(x, bool),
    "fun": callable,
    "str": lambda x: isinstance(x, str),
    "date": lambda x: isinstance(x, date),
    "datetime": lambda x: isinstance(x, datetime),
}


VALUE_PARSINGS = {
    "int": (r"^\d+$", int),
    "float": (r"^\d+.\d+$", float),
    "datetime": (RE_PATERNS['datetime'], to_datetime),
    "date": (RE_PATERNS['date'], to_date),
    "bool": (r"^True$|^False$", string_to_bool),
    "attribute": (RE_PATERNS["attr_shrt"], to_attribute)
}


def def_attribute(name: str, preset: str) -> Attribute:
    return Attribute(name, PREMADE_DOMAINS[preset])


def str_to_attribute(string: str) -> Optional[Attribute]:
    match = re.search(r"(\w*)\[(\w*)\]", string)
    if match:
        return Attribute(match.group(1), PREMADE_DOMAINS[match.group(2)])
    else:
        return None


# --------------- TABLE


def row_to_str(row: Row, col_width: int) -> str:
    res = ""
    for value in row:
        res += lrpad(str(value), col_width) + "|"
    return res[:-1]


def row_to_csv(row: Row, delimeter: str = ";") -> str:
    res = ""
    for index, value in enumerate(row):
        res += str(value)
        if index < len(row) - 1:
            res += delimeter
    return res


class Table:
    def __init__(self, name: str,
                 col_definitions: List[ColDefinition]):

        attributes = []
        defs = []
        keys = set()
        def_indexes = []

        for index, (attribute, key, def_value) in enumerate(col_definitions):
            attributes.append(attribute)

            if def_value is not None:
                defs.append(def_value)
                def_indexes.append(index)
            if key:
                keys.add(attribute.name)

        def row_transformer(row: Row, table_context: Table):
            new_row = []
            i = 0
            size = 0
            def_i = 0
            while i < len(row):
                if def_i < len(def_indexes) and def_indexes[def_i] == size:
                    def_value = defs[def_i]
                    new_row.append(get_def_value(def_value, table_context))
                    def_i += 1
                else:
                    new_row.append(row[i])
                    i += 1
                size += 1

            return new_row

        self.name = name
        self.attributes = attributes
        self.rows: List[Row] = []
        self.row_transformer = row_transformer
        self.cols = {}

        self.col_definitions = col_definitions

        self.id_counter = 0

        for index, attribute in enumerate(attributes):
            self.cols[attribute.name] = index

        self.keys = keys
        self.key_vals: DefaultDict[str, Set[Any]] = defaultdict(set)

    def copy_table(self) -> 'Table':
        return Table(self.name + "_copy", self.col_definitions)

    def check_row(self, row: Row) -> Optional[Row]:

        res_row = []
        if len(row) != len(self.attributes):
            return None

        for table_attr, row_value in zip(self.attributes, row):
            if not table_attr.domain.contains(row_value):
                return None
            else:
                res_row.append(table_attr.domain.cast(row_value))
        return tuple(res_row)

    def add_row(self, in_row: Row) -> None:

        if len(in_row) != len(self.cols):
            row = self.row_transformer(in_row, self)
        else:
            row = in_row
        casted_row = self.check_row(row)

        if casted_row is not None:
            for key in self.keys:
                if casted_row[self.cols[key]] in self.key_vals[key]:
                    raise RuntimeError("Key already present")
            self.rows.append(casted_row)
            for key in self.keys:
                self.key_vals[key].add(row[self.cols[key]])
            self.id_counter += 1
        else:
            raise RuntimeError(f"Invalid row: {row} for {self.name}")

    def remove_row(self, row: Row) -> None:
        i = 0
        while i < len(self.rows):
            if self.rows[i] == row:
                self.rows.pop(i)
                break
            i += 1

    def remove_rows(self, rows: List[Row]) -> None:
        for row in rows:
            self.remove_row(row)

    def __getitem__(self, colname: str) -> int:
        return self.cols[colname]

    def __str__(self) -> str:

        col_width = 20
        total_width = len(self.attributes) * col_width + \
            len(self.attributes) - 1
        divider = "\n" + (total_width) * "-" + "\n"

        res = "\n" + lrpad(self.name, total_width)
        res += "\n\n"
        res += row_to_str(tuple([("*" if x.name in self.keys else "") +
                                 str(x) for x in self.attributes]), 20)

        res += divider

        for row in self.rows:
            res += row_to_str(row, 20)
            res += divider

        return res


# --------------- DEF COLS


def create_def_cols(attributes: List[Attribute]) -> List[ColDefinition]:
    return [(x, False, None) for x in attributes]


# --------------- TABLE IO


def load_table_from(file_path: str) -> Table:
    with open(file_path, "r", encoding="utf-8") as f:

        [table_name, col_count] = next(f).strip().split(";")

        col_defs = []

        for _ in range(int(col_count)):
            [col_name, col_key, col_def] = next(f).strip().split(";")
            attribute_parsed = str_to_attribute(col_name)
            if attribute_parsed:
                col_defs.append(
                    (attribute_parsed, string_to_bool(col_key), col_def if col_def != "None" else None))
            else:
                RuntimeError("Unable to parse attribute")
        table = Table(table_name, col_defs)

        for _, row in enumerate(f):
            row_values = row.strip().split(";")
            table.add_row(tuple(row_values))

        return table


def write_table_to(table: Table, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(table.name + ";" + str(len(table.cols)) + "\n")

        for attribute, key, def_value in table.col_definitions:
            f.write(row_to_csv(tuple([str(attribute), key, def_value])) + "\n")

        for index, row in enumerate(table.rows):
            f.write(row_to_csv(row))
            if index < len(table.rows) - 1:
                f.write("\n")


def list_tables(storage_path: str) -> List[str]:

    res = []

    for _, _, files in os.walk(storage_path, topdown=False):
        res += [os.path.splitext(f)[0] for f in files]

    return res


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


def get_def_value(def_value: str, table_context: Table):
    if def_value == "%":
        return table_context.id_counter
    else:
        return def_value


def create_table_from_cols(name: str, cols: List[ColDefinition]) -> Table:
    return Table(name, cols)


# --------------- FACTORIES


def create_table_factory(name: str, cols: List[ColDefinition]) -> TableAction:
    def create_table(table: Table) -> Table:
        new_table = create_table_from_cols(name, cols)
        return new_table
    return create_table


def selection_factory(predicate: Predicate) -> TableAction:
    def selection(table: Table) -> Table:
        new_table = table.copy_table()
        new_table.name = "selection_table"

        for row in table.rows:
            if predicate(row, table):
                new_table.add_row(row)
        return new_table
    return selection


def projection_factory(row_functions: List[Callable[[Row], Any]], attributes: List[Attribute]) -> TableAction:

    def projection(table: Table) -> Table:
        new_attributes = []

        for attr in attributes:
            if attr not in table.cols:
                raise RuntimeError("Attribute not in table")
            new_attributes.append(table.attributes[table.cols[attr.name]])

        new_table = Table("projection_table", create_def_cols(new_attributes))

        for row in table.rows:
            new_row = []
            for attr in attributes:
                new_row.append(row[table.cols[attr.name]])
            new_table.add_row(tuple(new_row))
        return new_table

    return projection


def rename_factory(name: str) -> TableAction:
    def rename(table: Table) -> Table:
        table.name = name
        return table
    return rename


def insert_factory(function: TableFunction) -> TableAction:

    def insert(table: Table) -> Table:
        rows = function(table)
        for row in rows:
            table.add_row(row)
        return table

    return insert


def join_factory(type: str, left: TableAction, right: TableAction, condition) -> TableAction:
    def join(table: Table) -> Table:
        left_table = left(table)
        right_table = right(table)

        final_table = None

        if condition:
            raise RuntimeError("Condition is not yet supported")
        same_attributes = []
        final_attributes = left_table.attributes[:]
        right_indexes = []
        for r_attribute in right_table.attributes:
            if r_attribute.name in left_table.cols:
                same_attributes.append(r_attribute)
            else:
                right_indexes.append(right_table.cols[r_attribute.name])
                final_attributes.append(r_attribute)

        def final_transformer(left: Row, right: Row):
            return tuple(list(left) + [right[index] for index in right_indexes])

        final_table = Table(
            "joined_table", create_def_cols(final_attributes))

        for l_row in left_table.rows:
            for r_row in right_table.rows:
                add = True
                for attr in same_attributes:
                    if l_row[left_table.cols[attr.name]] != r_row[right_table.cols[attr.name]]:
                        add = False
                        break
                if add:
                    final_table.add_row(final_transformer(l_row, r_row))

        return final_table

    return join


def print_factory() -> TableAction:
    def print_tbl(table: Table) -> Table:
        print(table)
        return table
    return print_tbl


def nothing_factory() -> TableAction:
    def nothing(table: Table) -> Table:
        return table
    return nothing


def sort_factory(attributes: List[str]) -> TableAction:
    def sort(table: Table) -> Table:
        table.rows.sort(key=lambda x: [x[table.cols[attr]]
                                       for attr in attributes])
        return table
    return sort


# --------------- QUERY PARSING


def dot(a: TableAction, b: TableAction) -> TableAction:
    def dot_joined(table: Table) -> Table:
        return a(b(table))
    return dot_joined


def make_tuple(x: Any, y: Any) -> Row:
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
    "sort": (1, "*"),
    "nothing": 0,
}


DB_FUNCTIONS = [x for x in DB_FUNCTIONS_ARITIES]
OPERATORS = [x for x in OPERATORS_LEVELS]

Token = Tuple[str, Any]
ParsedToken = Tuple[str, str, Any]


class Variable:
    def __init__(self, var_type: str, name: str):
        self.var_type = var_type
        self.name = name
        self.tag = "var"

    def get(self) -> ParsedToken:
        return ("var", self.var_type, self.name)

    def __str__(self) -> str:
        return (f"{self.var_type} {self.name}")


class Constant:
    def __init__(self, val_type: str, value: Any):
        self.val_type = val_type
        self.value = value
        self.tag = "constant"

    def get(self) -> ParsedToken:
        return ("val", self.val_type, self.value)

    def __str__(self) -> str:
        return str(self.value) + " :: " + self.val_type

    def __repr__(self) -> str:
        return "Constant " + str(self)


ExpressionAtom = Union[Variable, Constant]


def result_to_val_tuple(result: Any) -> Tuple[str, str, Any]:
    val_type = "any"
    for name, tester in VAL_TESTERS.items():
        if tester(result):
            val_type = name
            break
    return "val", val_type, result


Operation = Any


def build_row_expression(left: ParsedToken, right: ParsedToken, operation: Operation) -> ParsedToken:
    left_state, left_type, left_val = left
    right_state, right_type, right_val = right

    if left_state == "val" and right_state == "val":
        if left_type != "row_function" and right_type != "row_function":
            return result_to_val_tuple(operation(left_val, right_val))

    def expression(row: Row, table: Table) -> ParsedToken:
        left_final_value = None
        if left_state == "var":
            left_final_value = row[table.cols[left_val]]
        elif left_state == "val":
            if left_type == "row_function":
                left_final_value = left_val(row, table)
            else:
                left_final_value = left_val

        right_final_value = None

        if right_state == "var":
            right_final_value = row[table.cols[right_val]]
        elif right_state == "val":
            if right_type == "row_function":
                right_final_value = right_val(row, table)
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


def get_special_chars() -> Dict[str, str]:

    res = {}

    for key in BINARY_OPERATIONS:
        res[key] = "operator"

    for key, (start, end) in ENCLOSINGS.items():
        res[start] = key
        res[end] = key

    return res


def tokenize_input(string: str) -> List[Token]:

    tokens: List[Token] = []

    word_buffer = ''
    quote_started = False

    special_chars = get_special_chars()

    i = 0

    while i < len(string):
        c = string[i]

        if starts_with(string, "'", i):
            if word_buffer and not quote_started:
                print(tokens)
                raise RuntimeError("Error occured whilte parsing text, check quotations" +
                                   " marks")
            if quote_started:
                tokens.append(("value", word_buffer))
                word_buffer = ""
            quote_started = not quote_started
            i += 1
            continue

        if quote_started:
            word_buffer += c
            i += 1
            continue

        special_char = None
        for s_char in special_chars:
            if starts_with(string, s_char, i):
                if quote_started:
                    raise RuntimeError(
                        "Quotes not ended properly, check quotation marks")
                if word_buffer:
                    tokens.append(tokenize_word(word_buffer))
                    word_buffer = ""
                tokens.append((special_chars[s_char], s_char))
                special_char = s_char
                break
        if special_char:
            i += len(special_char)
            continue

        if c == " ":
            if word_buffer:
                tokens.append(tokenize_word(word_buffer))
                word_buffer = ""
        else:
            word_buffer += c

        i += 1

    if word_buffer:
        tokens.append(tokenize_word(word_buffer))
        word_buffer = ""

    if quote_started:
        raise RuntimeError("Quotes not ended properly, check quotation marks")

    return tokens

    # CHECK ARITY
    arity = DB_FUNCTIONS_ARITIES[function_name]


# --------------- PARSING

def check_args(function_name: str, args: Args) -> bool:

    args_num = len(args)

    arity = DB_FUNCTIONS_ARITIES[function_name]

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


def parse_miniblock(tokens: List[Token]) -> ExpressionAtom:

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
        arg_tokens = [parse_miniblock([t]) for t in tokens[1:]]
        function = None

        if not check_args(head_val, arg_tokens):
            raise RuntimeError(f"Invalid arguments near {head_val}")
        arg_num = len(arg_tokens)

        if head_val == "project":
            pass
            #function = projection_factory([x.name for x in arg_tokens])

        elif head_val == "sort":
            function = sort_factory([x.name for x in arg_tokens])

        elif head_val == "load":
            if arg_tokens[0].tag == 'var':
                function = load_table_factory(storage_path, arg_tokens[0].name)
            else:
                print(arg_tokens[0].tag)
        elif head_val == "save":
            table_name = None
            if len(arg_tokens) == 1:
                table_name = arg_tokens[0].name
            function = save_table_factory(storage_path, table_name)

        elif head_val == "create":
            attributes = []

            for i in range(1, len(arg_tokens)):
                attribute, key, def_value = None, False, None
                package = arg_tokens[i].get()[2]
                if isinstance(package, tuple):
                    attribute = package[0]
                    for i in range(1, len(package)):
                        second = package[i]
                        if second == "key":
                            key = True
                        else:
                            def_value = second
                else:
                    attribute = package
                attributes.append((attribute, key, def_value))
            function = create_table_factory(arg_tokens[0].name, attributes)

        elif head_val == "insert":
            insert_argument = arg_tokens[0]
            if isinstance(insert_argument, Constant):
                if insert_argument.val_type == "tuple":
                    function = insert_factory(
                        lambda x: [insert_argument.value])
                elif insert_argument.val_type == "fun":
                    function = insert_factory(
                        lambda x: insert_argument.value(x).rows)
                else:
                    function = insert_factory(
                        lambda x: [[insert_argument.value]])
        elif head_val == "select":
            function = selection_factory(arg_tokens[0].get()[2])

        elif head_val == "print":
            function = print_factory()

        elif head_val == "nothing":
            function = nothing_factory()

        elif head_val == "rename":
            if arg_tokens[0].tag == "var":
                function = rename_factory(arg_tokens[0].name)

        elif head_val == "join":
            if arg_tokens[0].tag == 'var':
                left = arg_tokens[1 if arg_num == 3 else 2].get()[2]
                right = arg_tokens[2 if arg_num == 3 else 3].get()[2]
                condition = None if arg_num == 3 else arg_tokens[1].get()[2]
                function = join_factory(
                    arg_tokens[0].name, left, right, condition)

        if function is None:
            raise RuntimeError("Something went wrong near " + head_val)

        return Constant("table_transform", function)

    else:
        raise RuntimeError("Invalid header head " + head_type)


def parse_tokens(tokens: List[Token]) -> ExpressionAtom:

    brackets = []
    bracket_stack = []

    for index, (token_type, token_value) in enumerate(tokens):
        if token_type == "parentheses":
            if token_value == "(":
                bracket_stack.append(index)
            else:
                if not bracket_stack:
                    raise RuntimeError("Missing starting bracket")
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

                    left_parsed = parse_tokens(tokens[:index]).get()
                    right_parsed = parse_tokens(tokens[index + 1:]).get()

                    e_state, e_type, e_val = build_row_expression(
                        left_parsed, right_parsed, BINARY_OPERATIONS[token_value])

                    return Constant(e_type, e_val)

    return parse_miniblock(tokens)


# --------------- APPLICATION


def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Execute commands on database")

    parser.add_argument('-ls', '--list', action="store_const", const=True,
                        help="If set lists present tables")

    parser.add_argument('-s', '--session', action="store_const", const=True,
                        help="If set starts session")

    parser.add_argument('-i', '--isolated', action="store_const", const=True,
                        help="If set starts session in isolation from context")

    parser.add_argument('query', nargs="?", type=str, help="the main input")

    return parser.parse_args(sys.argv[1:])


def execute_query(query: str, isolation: bool) -> None:
    tokens = tokenize_input(query)
    tree = parse_tokens(tokens)

    if isinstance(tree, Constant) and tree.val_type in ("table_transform", "fun"):
        function = tree.get()[2]

        if not isolation:
            if os.path.exists(storage_path + "\\current.tbl"):
                load_current = load_table_factory(storage_path, "current")
                function = dot(function, load_current)
            save_current = save_table_factory(storage_path, "current")

            function = dot(save_current, function)
        print(function(None))
    else:
        print(tree)


if True:
    # pass
    tokenize_input = create_loud_function(tokenize_input)
    # parse_tokens = create_loud_function(parse_tokens)
    # starts_with = create_loud_function(starts_with)

if __name__ == "__main__":
    args = get_args()

    storage_path = get_local_storage()

    if not os.path.exists(storage_path):
        os.mkdir(storage_path)

    if args.list:
        print(list_tables(storage_path))

    query_que: Deque[str] = deque()
    if args.query:
        query_que.append(args.query)
    else:
        user_input = input(">>> ")
        if user_input:
            query_que.append(user_input)
    while query_que:
        query = query_que.popleft()

        execute_query(query, args.isolated)

        if args.session:
            user_input = input(">>> ")
            if user_input:
                query_que.append(user_input)
