from typing import (List, Any, Set,
                    Optional, Tuple, Callable,
                    Dict, Deque, DefaultDict, Union, Literal)

from videbugging import create_loud_function, create_louder_function

import os
import sys
import re
import argparse
import traceback
import logging
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
    "braces": ('{', "}")
}


def dateformat_to_regex(string: str) -> str:
    string = string.replace("%Y", r"\d{4}")
    string = string.replace("/", r"/")
    string = string.replace(".", r"\.")

    for char in "mdHMS":
        string = string.replace(f"%{char}", r"\d{2}")

    return string


DATE_FORMATS = [(x, dateformat_to_regex(x)) for x in [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H",
    "%Y-%m-%d",
    "%Y/%m/%d/%H/%M/%S",
    "%Y/%m/%d/%H/%M",
    "%Y/%m/%d/%H",
    "%Y/%m/%d",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y %H",
    "%d.%m.%Y",
]]

RE_PATERNS = {
    "attr_shrt": r"(\w+)\[(\w+)]",
    "attr_lng": r"(\w+)\[(\w+)]\[([\s\S]*)\]",
    "datetime": "|".join([x[1] for x in DATE_FORMATS]),
    "date": "|".join([x[1] for x in DATE_FORMATS]),
    "null": r"None"
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


def lpad(string: str, size: int, char: str = " ") -> str:
    padding = size - len(string)
    if padding <= 0:
        return string
    else:
        return char * padding + string


def rpad(string: str, size: int, char: str = " ") -> str:
    padding = size - len(string)
    if padding <= 0:
        return string
    else:
        return string + char * padding


def mlrow_to_slrows(strings: List[str]) -> List[List[str]]:
    cols_split = [s.splitlines() for s in strings]
    rows_size = [len(x) for x in cols_split]
    rows = [([0] * len(strings))
            for i in range(max(rows_size))]
    for index_col, col_string in enumerate(cols_split):
        for index_row, row in enumerate(col_string):
            rows[index_row][index_col] = row

    return rows


def str_map(lst: List[Any]) -> List[str]:
    return [str(x) for x in lst]


# --------------- TYPING


class TypeVariable:
    def __init__(self, name: str):
        self.name = name

    def get(self, context: Dict[str, str]):
        if self.name not in context:
            raise ValueError("Variable not defined")
        return context[self.name]


class TypeConst:
    def __init__(self, value: str):
        self.value = value

    def get(self, context: Dict[str, str]):
        return self.value

    def __str__(self) -> str:
        return str(self.value)


def find_type_gcd(left: str, right: str) -> Optional[str]:
    if left == right:
        return left
    if left in DOMAIN_INHERITANCE[right]:
        return right
    if right in DOMAIN_INHERITANCE[left]:
        return left
    return None


# find_type_gcd = create_loud_function(find_type_gcd)


def resolve_type_in_context(type_vars: Dict[str, str],
                            type_tuple: Tuple[str, str]) -> Optional[Tuple[str, str]]:
    type_val, type_name = type_tuple

    if type_name in type_vars:
        type_val = find_type_gcd(type_val, type_vars[type_name])

    if not type_val:
        return None

    type_vars[type_name] = type_val
    return type_val


# resolve_type_in_context = create_loud_function(resolve_type_in_context)


class TypeMask:
    def __init__(self, left: Tuple[str, str], right: Tuple[str, str],
                 output: Tuple[str, str]):
        self.left = left
        self.right = right
        self.output = output

        self.type_vars = {}

        if (not resolve_type_in_context(self.type_vars, left) or
                    not resolve_type_in_context(self.type_vars, right) or
                    not resolve_type_in_context(self.type_vars, output)
                ):
            raise RuntimeError(f"Invalid mask type: {str(self)}")

    def check(self, actual_left: str, actual_right: str) -> Optional[str]:
        type_vars = dict(self.type_vars)

        new_left = (actual_left, self.left[1])
        new_right = (actual_right, self.right[1])

        if (not resolve_type_in_context(type_vars, new_left) or
                not resolve_type_in_context(type_vars, new_right)):
            return None
        else:
            return type_vars[self.output[1]]

    def __str__(self):
        return (f"{self.left[0]} {self.left[1]} -> " +
                f"{self.right[0]} {self.right[1]} -> " +
                F"{self.output[0]} {self.output[1]}")


class TypeRelation:
    def __init__(self, left: Any, right: Any, type_mask: TypeMask):
        self.left = left
        self.right = right
        self.type_mask = type_mask

    def get(self, context: Dict[str, str]) -> Optional[str]:
        left_type = self.left.get(context)
        right_type = self.right.get(context)

        output_type = self.type_mask.check(left_type, right_type)

        if not output_type:
            raise RuntimeError(f"Invalid types {left_type} and " +
                               f"{right_type} are not supported under " +
                               f"the type mask {str(self.type_mask)}")
        return output_type

    def __str__(self):
        return str(self.type_mask)


Typing = Union[TypeConst, TypeVariable, TypeRelation]


class TypedFunction:
    def __init__(self, function: Callable[[Any], Any], typing: Typing):
        self.function = function
        self.typing = typing

    def get_type(self, context: Dict[str, str]):
        return self.typing.get(context)

    def __repr__(self) -> str:
        return f"TypedFunction {str(self.typing)}"


TYPE_TREE = {
    "any": [],
    "num": ["any"],
    "int": ["num"],
    "float": ["int"],
    "fun": ["any"],
    "row_function": ["fun"],
    "table_transform": ["fun"],

    "bool": ["any"],
    "str": ["any"],
    "date": ["any"],
    "datetime": ["any"],
}

TYPE_TREE['null'] = [x for x in TYPE_TREE]


def transitive_closure(dictionary: [Any, Any]):

    res: DefaultDict[Set[str]] = defaultdict(set)

    for key, vals in dictionary.items():
        res[key] |= set(dictionary[key])
        for val in vals:
            res[key] |= set(dictionary[val])

    if res == dictionary:
        return res
    else:
        return transitive_closure(res)


DOMAIN_INHERITANCE = transitive_closure(TYPE_TREE)

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


def str_to_datetime(value: str) -> Optional[datetime]:
    for form, regex in DATE_FORMATS:
        if re.match(regex, value):
            return datetime.strptime(value, form)
    return None


def to_datetime(value: Any) -> DateTime:
    if isinstance(value, str):
        parsed = str_to_datetime(value)
        if not parsed:
            raise RuntimeError("Invalid date format")
        return parsed
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
        parsed = str_to_datetime(value)
        if not parsed:
            raise RuntimeError("Invalid date format")
        return parsed.date()
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


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and re.match(RE_PATERNS['null'], value):
        return True
    else:
        return False


def to_null(value: Any) -> None:
    if is_null(value):
        return None
    else:
        raise ValueError("The value is not none")


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
    "datetime": AttributeDomain("datetime", to_datetime),
    "null": AttributeDomain("null", to_null)
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
    "null": lambda x: x is None
}


VALUE_PARSINGS = {
    "int": (r"^\d+$", int),
    "float": (r"^\d+.\d+$", float),
    "datetime": (RE_PATERNS['datetime'], to_datetime),
    "date": (RE_PATERNS['date'], to_date),
    "bool": (r"^True$|^False$", string_to_bool),
    "attribute": (RE_PATERNS["attr_shrt"], to_attribute),
    "null": (RE_PATERNS["null"], to_null)
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


def row_to_str(row: Row, seps: Tuple[str, str, str], *col_widths) -> str:
    left, middle, right = seps
    res = left
    for index, value in enumerate(row):
        res += rpad(str(value),
                    col_widths[index % len(col_widths)])
        if index < len(row) - 1:
            res += middle
    return res + right


def row_to_csv(row: Row, delimeter: str = ";") -> str:
    res = ""
    for index, value in enumerate(row):
        res += str(value)
        if index < len(row) - 1:
            res += delimeter
    return res


class Column:
    def __init__(self, attribute: Attribute, is_key: bool,
                 default_value: Optional[str], optional: bool = True):
        self.attribute = attribute
        self.is_key = is_key
        self.optional = optional and not is_key
        self.default_value = default_value

    def get_name(self):
        return self.attribute.name

    def get_type(self):
        return self.attribute.domain.name

    def copy(self):
        return Column(Attribute(self.get_name(), self.attribute.domain),
                      self.is_key,
                      self.default_value)

    def __str__(self):
        return f"{'*' if self.is_key else ''}{str(self.attribute)}"


Key = Tuple[Any, ...]


class TableIndex:
    def __init__(self):
        self.index = {}

    def add(self, key: Key, row: Row):
        self.index[key] = row

    def remove(self, key: Key):
        del self.index[key]


class Table:
    def __init__(self, name: str,
                 cols: List[Column]):

        self.name = name
        self.rows: List[Row] = []

        self.cols = cols
        self.col_to_index = {}
        for index, col in enumerate(cols):
            self.col_to_index[col.get_name()] = index

        self.keys = set()
        self.primary_keys = set()
        self.key_mask = []
        for index, col in enumerate(cols):
            if col.is_key:
                self.keys.add(col.attribute.name)
                self.key_mask.append(index)

        self.def_values = []
        self.def_indexes = []
        for index, col in enumerate(cols):
            if col.default_value is not None:
                self.def_values.append(col.default_value)
                self.def_indexes.append(index)

        self.id_counter = 0
        self.primary_index = TableIndex()

    def copy_table(self) -> 'Table':
        return Table(self.name + "_copy", [x.copy() for x in self.cols])

    # COLS

    def rename_col(self, index: int, new_name: str) -> None:

        if self.cols[index].get_name() == new_name:
            return

        if new_name in self.col_to_index:
            raise RuntimeError("The name of the column is already in use")

        old_name = self.cols[index].get_name()

        self.cols[index].attribute.name = new_name
        self.col_to_index[new_name] = self.col_to_index[old_name]
        del self.col_to_index[old_name]

        if old_name in self.keys:
            self.keys.remove(old_name)
            self.keys.add(new_name)

    # KEYS

    def get_row_key(self, row: Row) -> Key:
        return tuple([row[i] for i in self.key_mask])

    def key_already_present(self, key: Key) -> bool:
        return len(self.keys) != 0 and key in self.primary_keys

    def update_key(self, key: Key, old_key: Optional[Key] = None) -> None:
        if len(self.keys) != 0:
            if old_key is not None and old_key in self.primary_keys:
                self.primary_keys.remove(old_key)
            self.primary_keys.add(key)

    # ROW MANIPULATION

    def cast_row(self, row: Row) -> Row:
        res_row = []
        if len(row) != len(self.cols):
            return None

        for col, row_value in zip(self.cols, row):

            val_is_null = is_null(row_value)

            if val_is_null and not col.optional:
                raise ValueError(f"None value given to non-optional " +
                                 "column: {str(col)}")
            if not val_is_null and not col.attribute.domain.contains(row_value):
                raise ValueError(f"The value {row_value} is outside the " +
                                 f"columns {str(col)} domain")
            else:
                if val_is_null:
                    res_row.append(None)
                else:
                    res_row.append(col.attribute.domain.cast(row_value))
        return tuple(res_row)

    def add_row(self, in_row: Row) -> None:

        if len(in_row) != len(self.col_to_index):
            row = self.row_transformer(in_row, self)
        else:
            row = in_row

        casted_row = self.cast_row(row)
        row_key = self.get_row_key(casted_row)

        if self.key_already_present(row_key):
            raise ValueError("Key already present")

        self.rows.append(casted_row)
        self.update_key(row_key)

        self.id_counter += 1

    def find_row(self, row: Row) -> Optional[int]:
        index = None
        if row_key in self.primary_keys:
            row_key = self.get_row_key(row)
            i = 0
            while i < len(self.rows):
                if self.get_row_key(self.rows[i]) == row_key:
                    index = i
                    break
                i += 1
        return index

    def remove_row(self, row: Row, index: Optional[int] = None) -> None:

        if index is None:
            index = self.find_row(row)

        if index is not None:
            self.rows.pop(index)

    def update_row(self, org_row: Row, new_row: Row,
                   index: Optional[int] = None) -> None:

        if index is None:
            index = self.find_row(org_row)

        if index is not None:
            self.rows[index] = new_row
            new_key = self.get_row_key(new_row)
            old_key = self.get_row_key(org_row)
            if new_key != old_key:
                if self.key_already_present(new_key):
                    raise ValueError("Key already present")
                self.update_key(new_key, self.get_row_key(org_row))

    def remove_rows(self, rows: List[Row]) -> None:
        for row in rows:
            self.remove_row(row)

    def row_transformer(self, row: Row, table_context: 'Table') -> Row:

        new_row = []
        i = 0
        size = 0
        def_i = 0

        while i < len(row):
            if def_i < len(self.def_indexes) and self.def_indexes[def_i] == size:
                def_value = self.def_values[def_i]
                new_row.append(get_def_value(def_value, table_context))
                def_i += 1
            else:
                new_row.append(row[i])
                i += 1
            size += 1

        return new_row

    # OVERRIDING

    def __getitem__(self, colname: str) -> int:
        return self.col_to_index[colname]

    def __str__(self) -> str:

        str_rows = []

        cur_str_row = [f"{x.get_name()}\n[{x.get_type()}]" for x in self.cols]
        str_rows.append(cur_str_row)
        for row in self.rows:
            cur_str_row = [str(x) for x in row]
            str_rows.append(cur_str_row)

        total_rows = [mlrow_to_slrows(x) for x in str_rows]

        col_widths = [0] * len(self.cols)

        for total_row in total_rows:
            for line in total_row:
                col_widths = [max(col_widths[i], len(line[i]))
                              for i in range(len(line))]

        total_width = sum(col_widths) + len(self.cols) - 1

        divider = (total_width) * "━"
        dividers = ["─" * c_w for c_w in col_widths]
        double_dividers = ["═" * c_w for c_w in col_widths]

        res = "\n" + lrpad(self.name, total_width) + "\n"

        norm_row = ("│", "│", "│")
        div_start = ("┌", "┬", "┐")
        div_section = ("╞", "╪", "╡")
        div_middle = ("├", "┼", "┤")
        div_bottom = ("└", "┴", "┘")

        res += row_to_str(dividers, div_start, *col_widths) + "\n"

        row_count = len(total_rows)
        for index, total_row in enumerate(total_rows):
            for line in total_row:
                res += row_to_str(line, norm_row, * col_widths) + "\n"
            seps_set = div_middle
            div_set = dividers
            if index == 0:
                seps_set = div_section
                div_set = double_dividers
            if index == index == row_count - 1:
                seps_set = div_bottom

            res += row_to_str(div_set, seps_set, * col_widths) + "\n"

        return res


def get_typecontext_from_table(table: Table):
    context = {}

    for col in table.cols:
        context[col.get_name()] = col.get_type()

    return context


# --------------- DEF COLS


def create_def_cols(attributes: List[Attribute]) -> List[ColDefinition]:
    return [Column(x, False, None) for x in attributes]


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
                    Column(attribute_parsed, string_to_bool(col_key), col_def if col_def != "None" else None))
            else:
                RuntimeError("Unable to parse attribute")
        table = Table(table_name, col_defs)

        for _, row in enumerate(f):
            row_values = row.strip().split(";")
            table.add_row(tuple(row_values))

        return table


def write_table_to(table: Table, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(table.name + ";" + str(len(table.col_to_index)) + "\n")

        for col in table.cols:
            attribute, key, def_value = col.attribute, col.is_key, col.default_value
            f.write(row_to_csv(tuple([str(attribute), key, def_value])) + "\n")

        for index, row in enumerate(table.rows):
            f.write(row_to_csv(row))
            if index < len(table.rows) - 1:
                f.write("\n")


def delete_table(file_path: str) -> None:
    os.remove(file_path)


def list_tables(storage_path: str) -> List[str]:

    res = []

    for _, _, files in os.walk(storage_path, topdown=False):
        res += [os.path.splitext(f)[0] for f in files]

    return res


def get_local_storage() -> str:
    return os.path.dirname(os.path.realpath(sys.argv[0])) + "\\.viminidbstorage"


def get_db_storage(filename: str = None) -> str:
    addition = "" if filename is None else "\\" + filename
    return get_local_storage() + "\\tables" + addition


def get_scripts_storage(filename: str = None) -> str:
    addition = "" if filename is None else "\\" + filename
    return get_local_storage() + "\\scripts" + addition


def get_def_value(def_value: str, table_context: Table):
    if def_value == "%":
        return table_context.id_counter
    else:
        return def_value


def create_table_from_cols(name: str, cols: List[Column]) -> Table:
    return Table(name, cols)


# --------------- FACTORIES


def create_table_factory(name: str, cols: List[Column]) -> TableAction:
    def create_table(table: Table) -> Table:
        new_table = create_table_from_cols(name, cols)
        return new_table
    return create_table


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


def drop_table_factory(storage_path: str, table_name: str = None) -> TableAction:
    def drop(table: Table):
        if table:
            file_name = (table_name if table_name else table.name) + ".tbl"
            file_path = storage_path + "\\" + file_name
            delete_table(file_path)
        return None
    return drop


def selection_factory(predicate: 'TypedFunction') -> TableAction:
    def selection(table: Table) -> Table:
        new_table = table.copy_table()
        new_table.name = "selection_table"

        for row in table.rows:
            if predicate.function(row, table):
                new_table.add_row(row)
        return new_table
    return selection


def projection_factory(row_functions: List[TypedFunction]) -> TableAction:

    def projection(table: Table) -> Table:
        new_attributes = []

        for index, function in enumerate(row_functions):
            fun_type = function.get_type(get_typecontext_from_table(table))
            new_attributes.append(def_attribute(f"col_{index}", fun_type))

        new_table = Table("projection_table", create_def_cols(new_attributes))

        for row in table.rows:
            new_row = []
            for function in row_functions:
                new_row.append(function.function(row, table))
            new_table.add_row(tuple(new_row))
        return new_table

    return projection


def update_factory(condition: TypedFunction, row_functions: List[TypedFunction]) -> TableAction:

    def update(table: Table) -> Table:

        if len(table.cols) != len(row_functions):
            raise ValueError("The number of functions in update doesn't " +
                             f"match the number of columns: " +
                             f"{len(table.cols)} != {len(row_functions)}")

        for index, row in enumerate(table.rows):
            if condition.function(row, table):
                new_row = [f.function(row, table) for f in row_functions]
                table.update_row(row, tuple(new_row), index)

        return table

    return update


def remove_factory(condition: TypedFunction):

    def remove(table: Table) -> Table:
        i = 0
        while i < len(table.rows):
            if condition.function(table.rows[i], table):
                table.remove_row(table.rows[i], i)
            else:
                i += 1
        return table
    return remove


def rename_factory(name: str, attributes: Optional[List[str]]) -> TableAction:
    def rename(table: Table) -> Table:
        if name:
            table.name = name

        if attributes:
            for index, attribute in enumerate(attributes):
                table.rename_col(index, attribute)

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
        final_attributes = left_table.cols[:]

        right_indexes = []
        for r_attribute in right_table.attributes:
            if r_attribute.name in left_table.col_to_index:
                same_attributes.append(r_attribute)
            else:
                right_indexes.append(
                    right_table.col_to_index[r_attribute.name])
                final_attributes.append(r_attribute)

        def final_transformer(left: Row, right: Row):
            return tuple(list(left) + [right[index] for index in right_indexes])

        final_table = Table(
            "joined_table", create_def_cols(final_attributes))

        for l_row in left_table.rows:
            for r_row in right_table.rows:
                add = True
                for attr in same_attributes:
                    if l_row[left_table.col_to_index[attr.name]] != r_row[right_table.col_to_index[attr.name]]:
                        add = False
                        break
                if add:
                    final_table.add_row(final_transformer(l_row, r_row))

        return final_table

    return join


def table_prefix_attributes(table: Table):
    for i, col in enumerate(table.cols):
        table.rename_col(i, table.name + "_" + col.get_name())
    return table


def cartesian_factory(left: TableAction, right: TableAction, condition: TypedFunction = None) -> TableAction:
    def cartesian(table: Table):
        left_table = table_prefix_attributes(left(table))
        right_table = table_prefix_attributes(right(table))

        new_cols = left_table.cols + right_table.cols
        final_table = Table("cartesian_product", new_cols)

        for left_row in left_table.rows:
            for right_row in right_table.rows:
                new_row = tuple(list(left_row) + list(right_row))
                if not condition or condition.function(new_row, final_table):
                    final_table.add_row(new_row)
        return final_table
    return cartesian


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
        table.rows.sort(key=lambda x: [x[table.col_to_index[attr]]
                                       for attr in attributes])
        return table
    return sort


def script_factory(script_name: str, arguments: List[str]) -> TableAction:
    query = read_script(script_name)

    def script(table: Table) -> Table:
        try:
            group = execute_query(
                query, arguments, False, False, False, False)
        except:
            raise RuntimeError(f"Script {script_name} failed, there is , " +
                               "probably additional error output above")
        return group[0]
    return script


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


def in_function(value: Any, table: Table) -> TableAction:
    if len(table.cols) != 1:
        raise ValueError("Operator 'in' works only in tables " +
                         "with one column")
    for row in table.rows:
        if row[0] == value:
            return True
    return False


logic = TypeMask(("any", "a"), ("any", "a"), ("bool", "b"))
arithmetic = TypeMask(("num", "a"), ("num", "a"), ("num", "a"))


BINARY_OPERATIONS = {
    ".": (lambda x, y: dot(x, y), TypeMask(("fun", "a"), ("fun", "b"), ("fun", "c"))),
    ">>": (lambda x, y: dot(y, x), TypeMask(("fun", "a"), ("fun", "b"), ("fun", "c"))),
    ",": (lambda x, y: make_tuple(x, y), TypeMask(("any", "a"), ("any", "b"), ("tuple", "c"))),
    "=>": (lambda x, y: in_function(x, y(None)),
           TypeMask(("any", "a"), ("fun", "b"), ("bool", "c"))),
    "!>": (lambda x, y: not in_function(x, y(None)),
           TypeMask(("any", "a"), ("fun", "b"), ("bool", "c"))),
    "&&": (lambda x, y: x and y, logic),
    "||": (lambda x, y: x or y, logic),

    "==": (lambda x, y: x == y, logic),
    "!=": (lambda x, y: x != y, logic),
    "<=": (lambda x, y: x <= y, logic),
    ">=": (lambda x, y: x >= y, logic),
    "<": (lambda x, y: x < y, logic),
    ">": (lambda x, y: x > y, logic),

    "+": (lambda x, y: x + y, arithmetic),
    "-": (lambda x, y: x - y, arithmetic),
    "%": (lambda x, y: x % y, arithmetic),
    "*": (lambda x, y: x * y, arithmetic),
    "//": (lambda x, y: x // y, arithmetic),
    "/": (lambda x, y: x / y, TypeMask(("num", "a"), ("num", "a"), ("float", "a"))),
    "^": (lambda x, y: x ** y, arithmetic),


}

OPERATORS_LEVELS = {
    ".": 1,
    ">>": 1,
    ",": 2,
    "=>": 2,
    "!>": 2,

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
    "//": 6,
    "^": 7
}

DB_FUNCTIONS_ARITIES = {
    "nothing": 0,
    "print": 0,

    "create": (2, "*"),
    "load": 1,
    "save": (0, 1),
    "drop": (0, 1),
    "rename": (1, "*"),

    "project": (1, "*"),
    "select": 1,

    "insert": 1,
    "remove": 1,
    "update": (1, "*"),

    "sort": (1, "*"),
    "cartesian": (2, 3),
    "join": (3, 4),

    "script": (1, "*"),
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


def cast_to_row_function(token: ParsedToken) -> TypedFunction:
    t_state, t_type, t_val = token
    if t_state == "val":
        if t_type != "row_function":
            return TypedFunction(lambda row, table: t_val, TypeConst(t_type))
        else:
            return t_val
    if t_state == "var":
        return TypedFunction(lambda row, table: row[table.col_to_index[t_val]], TypeVariable(t_val))
    else:
        raise ValueError(f"{token} can't be parsed")


def get_token_type_object(token: ParsedToken) -> Optional[Typing]:
    token_state, token_type, token_val = token
    token_type_object = None
    if token_state == "val":
        if token_type == "row_function":
            token_type_object = token_val.typing
        else:
            token_type_object = TypeConst(token_type)
    elif token_state == "var":
        token_type_object = TypeVariable(token_val)
    return token_type_object


def get_token_value(token: ParsedToken, row: Row, table: Table) -> Optional[Any]:
    token_state, token_type, token_val = token
    token_final_value = None
    if token_state == "var":
        token_final_value = row[table.col_to_index[token_val]]
    elif token_state == "val":
        if token_type == "row_function":
            token_final_value = token_val.function(row, table)
        else:
            token_final_value = token_val
    return token_final_value


def build_row_expression(left: ParsedToken, right: ParsedToken, operation: Operation) -> ParsedToken:
    left_state, left_type, left_val = left
    right_state, right_type, right_val = right

    if left_state == "val" and right_state == "val":
        if left_type != "row_function" and right_type != "row_function":
            return result_to_val_tuple(operation[0](left_val, right_val))

    left_type_object = get_token_type_object(left)
    right_type_object = get_token_type_object(right)
    typing = TypeRelation(left_type_object, right_type_object, operation[1])

    def expression(row: Row, table: Table) -> ParsedToken:
        left_final_value = get_token_value(left, row, table)
        right_final_value = get_token_value(right, row, table)
        return operation[0](left_final_value, right_final_value)

    return "val", "row_function", TypedFunction(expression, typing)


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

    correct_arity = True

    if isinstance(arity, int):
        correct_arity &= args_num == arity
    elif isinstance(arity, tuple):
        lower_bound, upper_bound = arity
        if isinstance(lower_bound, int):
            correct_arity &= args_num >= lower_bound
        if isinstance(upper_bound, int):
            correct_arity &= args_num <= upper_bound

    if not correct_arity:
        raise RuntimeError("The number of arguments doesn't match the " +
                           f"functions arity, {args_num} not in {arity}")

    return correct_arity


def parse_miniblock(tokens: List[Token]) -> ExpressionAtom:

    db_storage = get_db_storage()
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
        tok_res = [x.get() for x in arg_tokens]
        tok_val = [x[2] for x in tok_res]
        function = None

        if not check_args(head_val, arg_tokens):
            raise RuntimeError(f"Invalid arguments near {head_val}")
        arg_num = len(arg_tokens)

        if head_val == "project":
            function = projection_factory(
                [cast_to_row_function(x) for x in tok_res])

        elif head_val == "script":
            function = script_factory(tok_val[0], str_map(tok_val[1:]))

        elif head_val == "update":
            function = update_factory(cast_to_row_function(tok_res[0]),
                                      [cast_to_row_function(x) for x in tok_res[1:]])

        elif head_val == "remove":
            function = remove_factory(
                cast_to_row_function(arg_tokens[0].get()))

        elif head_val == "sort":
            function = sort_factory(tok_val)

        elif head_val == "load":
            function = load_table_factory(db_storage, tok_val[0])

        elif head_val == "save":
            table_name = None
            if len(arg_tokens) == 1:
                table_name = tok_val[0]
            function = save_table_factory(db_storage, table_name)
        elif head_val == "drop":
            table_name = None
            if len(arg_tokens) == 1:
                table_name = tok_val[0]
            function = drop_table_factory(db_storage, table_name)

        elif head_val == "create":
            create_cols = []

            for package in tok_val[1:]:
                attribute, key, def_value = None, False, None
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
                create_cols.append(Column(attribute, key, def_value))
            function = create_table_factory(arg_tokens[0].name, create_cols)

        elif head_val == "insert":
            insert_argument = arg_tokens[0]
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
            function = selection_factory(tok_val[0])

        elif head_val == "print":
            function = print_factory()

        elif head_val == "nothing":
            function = nothing_factory()

        elif head_val == "rename":
            attribute_names = None
            if len(tok_val) > 1:
                attribute_names = tok_val[1:]
            function = rename_factory(tok_val[0], attribute_names)

        elif head_val == "join":
            if arg_tokens[0].tag == 'var':
                left = arg_tokens[1 if arg_num == 3 else 2].get()[2]
                right = arg_tokens[2 if arg_num == 3 else 3].get()[2]
                condition = None if arg_num == 3 else arg_tokens[1].get()[2]
                function = join_factory(
                    arg_tokens[0].name, left, right, condition)

        elif head_val == "cartesian":
            condition = tok_val[2] if len(tok_val) > 2 else None
            function = cartesian_factory(tok_val[0], tok_val[1], condition)

        if function is None:
            raise RuntimeError("Something went wrong near " + head_val)

        return Constant("table_transform", function)

    else:

        raise RuntimeError("Invalid header head " + head_type + ". The " +
                           "tokens were" + str(tokens))


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
        middle = [("parsed", parse_tokens(
            tokens[left + 1 - offset: right - offset]))]
        tokens = tokens[:left - offset] + middle + tokens[right + 1 - offset:]
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

    parser.add_argument('-t', '--type', action="store_const", const=True,
                        help="If set types the expression instead of execution")

    parser.add_argument('-r', '--reset_session', action="store_const", const=True,
                        help="If set, doesn't load context from last session")

    group = parser.add_mutually_exclusive_group()

    group.add_argument('-rs', '--record_script', type=str, help="Records " +
                       "the query into a script file with name")

    group.add_argument('-us', '--use_script', type=str,
                       help="If set uses script with the query name")

    parser.add_argument('--rec_raw', action='store_const', const='True',
                        help="If set, records @ arguments.")

    parser.add_argument('-v', '--verbose', action='store_const', const='True',
                        help="If set, shows debbuging information")

    parser.add_argument('main_args', nargs="*",
                        type=str, help="the main input")

    return parser.parse_args(sys.argv[1:])


def subtitute_args(query: str, query_args: List[str]):
    for i, arg in enumerate(query_args):
        match = re.search(r"@" + str(i + 1) + r"\|(\w+)@", query)
        if match:
            arg_type = match.group(1)
            if arg_type == 'str':
                arg = f"'{arg}'"

            query = re.sub(r"@" + str(i + 1) + r"\|(\w+)@", arg, query)
            continue

        match = re.search(r"@" + str(i + 1) + r"([\D$])", query)
        if match:
            query = re.sub(r"@" + str(i + 1) + r"([\D$])", arg + r"\1", query)
            continue

    return query


# subtitute_args = create_louder_function(subtitute_args)


def execute_query(query: str, query_args: List[str], type: bool,
                  load: bool, save: bool,
                  ret_original_query: bool = False) -> (
                      Tuple[Optional[Table], Optional[Any]]):
    original_query = query

    query = subtitute_args(query, query_args)
    ret_query = query if not ret_original_query else original_query
    tokens = tokenize_input(query)
    tree = parse_tokens(tokens)

    if not type and isinstance(tree, Constant) and tree.val_type in ("table_transform", "fun"):
        function = tree.get()[2]

        if load:
            if os.path.exists(storage_path + "\\current.tbl"):
                load_current = load_table_factory(storage_path, "current")
                function = dot(function, load_current)
        if save:
            save_current = save_table_factory(storage_path, "current")
            function = dot(save_current, function)
        return function(None), None, ret_query
    else:
        return None, tree, ret_query


def read_script(name: str) -> str:
    with open(get_scripts_storage(name + ".mndb"), "r", encoding="utf-8") as f:
        lines = []
        for l in f:
            lines.append(l)
        return " . ".join(lines)


def save_script(name: str, lines: List[str]) -> None:
    with open(get_scripts_storage(name + ".mndb"), "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    args = get_args()

    storage_path = get_local_storage()

    if not os.path.exists(storage_path):
        os.mkdir(storage_path)

    if not os.path.exists(get_db_storage()):
        os.mkdir(get_db_storage())

    if not os.path.exists(get_scripts_storage()):
        os.mkdir(get_scripts_storage())

    if args.verbose:
        tokenize_input = create_louder_function(tokenize_input)
        parse_tokens = create_louder_function(parse_tokens)
        build_row_expression = create_louder_function(build_row_expression)

    if args.list:
        print(list_tables(get_db_storage()))

    query_history = []

    query_que: Deque[str] = deque()

    if args.use_script:
        query_que.append(read_script(args.use_script))

    query = None
    query_args = args.main_args if args.main_args else []

    if not args.use_script and args.main_args:
        query = args.main_args[0]
        query_args = args.main_args[1:]

    if query:
        query_que.append(query)

    if not query_que:
        user_input = input(">>> ")
        if user_input:
            query_que.append(user_input)

    while query_que:
        query = query_que.popleft()

        load, save = True, True

        if args.isolated:
            load &= False
            save &= False

        if args.reset_session:
            load &= False

        try:
            table, additional, executed_query = execute_query(
                query, query_args, args.type, load, save, args.rec_raw)
            query_history.append(executed_query)
            if table:
                print(table)
            if additional:
                print(additional)
        except Exception as e:
            logging.error(traceback.format_exc())

        if args.session:
            user_input = input(">>> ")
            if user_input:
                query_que.append(user_input)

    if query_history and args.record_script:
        save_script(args.record_script, query_history)
