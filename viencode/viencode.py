from typing import Dict, List, Set, DefaultDict, Callable, Any, Tuple
from collections import defaultdict
from pprint import pprint
import json
import argparse
import sys

import re

DictData = Dict[str, str]
SoundDict = Dict[str, List[str]]

NUMBER_TO_SOUND_US = {
    "0": ["s", "z"],
    "1": ["t", "d", "θ", "ð"],
    "2": ["n"],
    "3": ["m"],
    "4": ["r"],
    "5": ["ɫ"],
    "6": ["tʃ", "dʒ", "ʃ", "ʒ"],
    "7": ["k", "ɡ"],
    "8": ["f", "v"],
    "9": ["p", "b"]
}

NUMBER_TO_SOUND_SK = {
    "0": ["s", "š", "z", "ž"],
    "1": ["t", "d", "ť", "ď"],
    "2": ["n", "ň"],
    "3": ["m"],
    "4": ["r", "ŕ"],
    "5": ["l", "ĺ", "ľ"],
    "6": ["j", "c", "č", "dz", "dž"],
    "7": ["k", "g", "h", "ch"],
    "8": ["f", "v", "w"],
    "9": ["p", "b"]
}


def load_file_us(file_name: str) -> DictData:
    res = {}
    with open(file_name, encoding="utf-8") as f:
        obj = json.load(f)
        dictionary = obj["en_US"][0]
        # for (word, pronounc), index in zip(dictionary.items(), range(1, 10)):
        for word, pronounc in dictionary.items():
            key = re.match(r'/(\w*)/', pronounc)
            res[key.group(1)] = word

    return res


def load_file_sk(file_name: str) -> DictData:
    res = {}
    with open(file_name, encoding="utf-8") as f:
        for l in f:
            regex_match = re.match(
                r'\d+\s\w\s+([\wáéíóôúýäčšžťľŕĺňď.,:;!?/\\\(\)\[\]\s"-/+]*)␞', l)

            if regex_match:
                for word in re.finditer(r'\b([\wáéíóôúýäčšžťľŕĺňď]+)\b', regex_match.group(1)):
                    key = word.group(1)
                    res[key] = key

    return res


def combine_words(words: List[List[str]]):
    return combine_words_acc(words, 0, [])


def combine_words_acc(words: List[List[str]], index: int, prefix: List[str]):
    if index == len(words):
        yield " ".join(prefix)
    else:
        for word in words[index]:
            prefix.append(word)
            yield from combine_words_acc(words, index + 1, prefix)
            prefix.pop()


def generate_addins(number: int) -> List[List[int]]:
    return generate_addins_acc(number, [], [])


def generate_addins_acc(number: int, prefix: List[int],
                        res: List[List[int]]) -> List[List[int]]:
    if number == 0:
        res.append(prefix[:])

    for i in range(1, number + 1):
        prefix.append(i)
        generate_addins_acc(number - i, prefix, res)
        prefix.pop()

    return res


def inverse_num_to_sound(sound_dict: SoundDict) -> Dict[str, str]:
    res = {}

    for num, sounds in sound_dict.items():
        for sound in sounds:
            res[sound] = num

    return res


def string_to_number(string: str, inversed_sound_dict: Dict[str, str]) -> str:
    string = string.lower()
    res = ""

    i = 0
    while i < len(string):
        char = string[i]

        if i < len(string) - 1:
            char_duo = char + string[i + 1]
            if char_duo in inversed_sound_dict:
                res += inversed_sound_dict[char_duo]
                i += 2
                continue

        if char in inversed_sound_dict:
            res += inversed_sound_dict[char]

        i += 1

    return res


def transform_dict(dictionary: Dict[Any, Any],
                   fun_key: Callable[[Any, Any], Any] = lambda key, val: key,
                   fun_val: Callable[[Any, Any], Any] = lambda key, val: val) -> Dict[Any, Any]:
    res = defaultdict(list)
    for key, val in dictionary.items():
        res[fun_key(key, val)].append(fun_val(key, val))

    return res


def get_substrings(string: str, lengths: List[int]) -> List[str]:
    i = 0
    res = []
    for length in lengths:
        word = string[i: i + length]
        res.append(word)
        i += length
    return res


def number_to_sentence(num: int, dict_number_to_word: DictData,):
    number = str(num)

    addins = generate_addins(len(number))
    addins.sort(key=lambda x: len(x))

    for variation in addins:
        words = get_substrings(number, variation)

        sentence = []
        valid = True
        for word in words:
            words_in_position = dict_number_to_word[word]
            sentence.append(words_in_position)
        if valid:
            yield from combine_words(sentence)


def highlight_chars(string: str, chars: List[str]):
    string = string.lower()
    res = ""

    i = 0
    while i < len(string):
        char = string[i]

        if i < len(string) - 1:
            char_duo = char + string[i + 1]
            if char_duo in chars:
                res += char_duo.upper()
                i += 2
                continue
        if char in chars:
            res += char.upper()
        else:
            res += char.lower()
        i += 1

    return res


def get_args() -> None:
    parser = argparse.ArgumentParser(
        description="Execute commands on folder of images")

    parser.add_argument(
        '-w', '--word', action='store_const', const=True, help="If set " +
        "shows number for the word at main input")

    parser.add_argument(
        '-p', '--perpage', type=int, help="Number of outputs per page")

    parser.add_argument(
        "-e", "--english", action="store_const", const=True, help="If set " +
        "uses english system instead of slovak"
    )

    parser.add_argument('main_input', type=str, help="the main input")

    return parser.parse_args(sys.argv[1:])


Args = Any


def get_preset(name: str) -> Tuple[Dict[str, str], List[str]]:
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    if name == "slovak":
        sk_inversed = inverse_num_to_sound(NUMBER_TO_SOUND_SK)
        sk_symbols = [x for x in sk_inversed]

        sk_num_to_word = load_file_sk(script_path + "\\sk.txt")
        sk_num_to_word = transform_dict(
            sk_num_to_word,
            lambda key, val: string_to_number(key, sk_inversed))
        return sk_inversed, sk_symbols, sk_num_to_word
    else:
        en_inversed = inverse_num_to_sound(NUMBER_TO_SOUND_US)
        en_symbols = []
        en_num_to_word = load_file_us(script_path + "\\en_US.json")
        en_num_to_word = transform_dict(
            en_num_to_word,
            lambda key, val: string_to_number(key, en_inversed))
        return en_inversed, en_symbols, en_num_to_word


if __name__ == "__main__":

    args = get_args()

    (inversed_number_to_sound,
     important_symbols,
     num_to_word) = get_preset("english" if args.english else "slovak")

    if not args.word:
        number = 0
        try:
            number = int(args.main_input)
        except:
            print(f"Input error: {args.main_input} is not a number")
            sys.exit(1)
        number_per_page = args.perpage if args.perpage else 10
        solution_generator = number_to_sentence(
            number, num_to_word)

        user_input = ""
        while not user_input:
            for word, _ in zip(solution_generator, range(0, number_per_page)):
                print(highlight_chars(word, important_symbols))
            user_input = input()
    else:
        decoded_number = string_to_number(
            args.main_input, inversed_number_to_sound)
        print(highlight_chars(args.main_input, important_symbols), decoded_number)
