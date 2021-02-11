from typing import List, Dict, Tuple

import pyperclip
import sys
import os
import argparse
import re

from pylatexenc.latex2text import LatexNodes2Text


def parse_latex(string: str) -> str:
    match = re.match(r"\\frac\{([\s\S]*)\}\{([\s\S]*)\}", string)
    string = re.sub(r"\\frac\{([\s\S]*)\}\{([\s\S]*)\}",
                    r"\\frac{(\1)}{(\2)}", string)
    print(string)
    return LatexNodes2Text().latex_to_text(string)


def make_equation(string: str) -> str:
    return "\\begin{equation}\\begin{split}" + string + "\\end{split}\\end{equation}"


def find_all(string: str, substring: str) -> List[int]:
    index = string.find(substring)
    res = []
    while index >= 0:
        res.append(index)
        index = string.find(substring, index + len(substring))
    return res


def one_d_collision(x1, x2, y1, y2):
    x_start = min(x1, x2)
    x_end = max(x1, x2)
    y_start = min(y1, y2)
    y_end = max(y1, y2)

    return (x_start <= y_start <= x_end or x_start <= y_end <= x_end or
            y_start <= x_start <= y_end or y_start <= x_end <= y_end)


SegmentDict = Dict[str, List[Tuple[int, int, str]]]


def remove_collisions(segment_dict: SegmentDict) -> SegmentDict:
    collisions = [("@@@", "@"), (">>>", ">")]

    for strong, weak in collisions:
        weak_finds = segment_dict[weak]
        strong_finds = segment_dict[strong]
        new_weak = []

        for weak_find in weak_finds:
            w_start, w_end, _, _ = weak_find
            adding = True

            for s_start, s_end, _, _ in strong_finds:
                if one_d_collision(s_start, s_end, w_start, w_end):
                    adding = False
                    break

            if adding:
                new_weak.append(weak_find)

        segment_dict[weak] = new_weak
    return segment_dict


def concat_dict(segment_dict: SegmentDict) -> [Tuple[int, int, "str"]]:
    res = []

    for _, lst in segment_dict.items():
        res += lst

    res.sort(key=lambda x: x[0])
    return res


Range1D = Tuple[int, int]


def split_to_segments(query: str) -> List[str]:
    function_strings = {
        "@@@": "math_segment",
        "@": "math_inline",
        ">>>": "code_segment",
        ">": "code_inline"
    }

    finds = {}

    for key, value in function_strings.items():
        finds[key] = [(x, x + len(key), value, len(key))
                      for x in find_all(query, key)]

    finds = remove_collisions(finds)

    all_segments = concat_dict(finds)

    connected_segments = []

    connected_segments_stack = []
    for segment in all_segments:
        seg_start, seg_end, seg_value, _ = segment

        last_value = None
        if len(connected_segments_stack) > 0:
            last_value = connected_segments_stack[-1][2]

        if last_value == seg_value:
            start_start, _, start_value, seg_limlen = connected_segments_stack.pop()
            connected_segments.append(
                (start_start, seg_end, start_value, seg_limlen))
        else:
            connected_segments_stack.append(segment)

    last_end = 0
    end = len(query)

    empty_ranges = []

    for seg_start, seg_end, seg_value, _ in connected_segments:
        if last_end != seg_start:
            empty_ranges.append((last_end, seg_start, "text", 0))
        last_end = seg_end
    if end != last_end:
        empty_ranges.append((last_end, end, "text", 0))

    return sorted(connected_segments + empty_ranges, key=lambda x: x[0])


def split_to_word_other(string: str) -> Tuple[str, str]:
    match = re.search("(\s)", string)

    if match:
        start, end = match.span()

        if start != 0:
            return string[:start], string[end:]

    return "", string


def build_code(substring: str) -> str:

    word, other = split_to_word_other(substring)

    return (f"\n```{word}\n" + other + "\n```\n")


def new(query: str) -> str:
    segments = split_to_segments(query)
    print(segments)
    result = ""

    for (start, end, segment_type, seg_mark_len) in segments:
        substring = query[start + seg_mark_len:end - seg_mark_len]

        if segment_type == "math_segment":
            result += ("\n```" +
                       parse_latex(make_equation(substring.replace("\n", "\\\\"))) + "```\n")
        elif segment_type == "math_inline":
            result += "`" + parse_latex(substring) + "`"
        elif segment_type == "code_segment":
            result += build_code(substring)
        else:
            result += substring
    return result


def old(query: str) -> str:
    result = ""

    for seg_index, segment in enumerate(query.split("@@@")):
        if seg_index % 2 == 0:
            for str_index, string in enumerate(segment.split("@")):
                if str_index % 2 == 0:
                    result += string
                else:
                    result += "`" + parse_latex(string) + "`"
        else:
            result += ("\n```" +
                       parse_latex(make_equation(segment)) + "```\n")
    return result


def get_args() -> None:
    parser = argparse.ArgumentParser(
        description="Execute commands on database")

    parser.add_argument('query', nargs="?", type=str, help="the main input")

    parser.add_argument('-ml', "--multiline", action="store_const", const=True,
                        help="If set accepts multiline text")

    return parser.parse_args(sys.argv[1:])


def get_input(multiline: bool) -> str:
    if multiline:
        lines = []
        while True:
            line = input("> ")
            if line:
                lines.append(line)
            else:
                break
        return '\n'.join(lines)
    else:
        return input("> ")


if __name__ == "__main__":

    args = get_args()

    session = not args.query
    exp_counter = 0

    while session or exp_counter == 0:
        query = args.query if args.query else get_input(args.multiline)

        result = new(query)

        # result = old(query)
        pyperclip.copy(result)

        result = result.replace("\n\n", "\n")
        print("\n\nRESULT: (It's also in your clipboard)")
        print(result)
        exp_counter += 1
