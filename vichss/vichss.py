from typing import Any

import sys
import argparse
import os

import chess
from stockfish import Stockfish
import re

import json

from pprint import pprint

Save = Any


def parse_string_to_move(string: str) -> str:
    command = string.strip()

    match = re.match(r"(\w)(\d)", command)
    if match:
        return match

    match = re.match(r"(\w)(\w)(\d)", command)
    if match:
        return match

    match = re.match(r"(\w)(\d)(\w)(\d)", command)
    if match:
        return match


def init_save_file(file_path: str) -> Save:
    with open(file_path, "w") as f:
        save_file = {
            "current_game": None,
            "stockfish_location": None
        }
        json.dump(save_file, f)
        return save_file


def load_save_file(file_path: str) -> Save:

    with open(file_path, "r") as f:
        save_file = json.load(f)
        return save_file


def write_save_file(file_path: str, save_file: Save) -> Save:
    with open(file_path, "w") as f:
        json.dump(save_file, f)
        return save_file


def get_fen_from_save(save: Save) -> str:
    return save["current_game"]["board_fen"]


def get_player_from_save(save: Save) -> str:
    return save["current_game"]["player_color"]


def get_player_from_fen(fen: str) -> str:
    match = re.search(r"\S*\s(\w)", fen)
    if match:
        return match.group(1)
    return "none"


Args = Any


def get_args() -> Args:

    parser = argparse.ArgumentParser(
        description="Chess with stockfish in terminal")

    parser.add_argument(
        '-r', '--reset',
        action='store_const', const=True,
        help="Resets the game")

    parser.add_argument(
        '-p', '--print',
        action='store_const', const=True,
        help="Prints the current game"
    )

    parser.add_argument(
        '-sl', '--stockfish_level',
        help="Sets the stockfish level"
    )

    parser.add_argument(
        '--stockfish_location',
        help="Sets the stockfish location"
    )

    parser.add_argument(
        'move',
        default="",
        nargs="?",
        help="move you want to play")

    return parser.parse_args(sys.argv[1:])


def end(exit_code, save: Save, file_name: str) -> None:
    if exit_code == 0:
        write_save_file(file_name, save)
    sys.exit(exit_code)


if __name__ == "__main__":

    args = get_args()

    config_name = "\\.vichsssave"
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    config_path = script_path + config_name

    if not os.path.exists(config_path) or args.reset:
        init_save_file(config_path)

    save_file = load_save_file(config_path)

    if not save_file["current_game"]:
        new_board = chess.Board()
        save_file["current_game"] = {
            "board_fen": new_board.fen(),
            "player_color": "w",
            "stockfish_level": 3
        }

    board = chess.Board(get_fen_from_save(save_file))

    player = get_player_from_fen(board.fen())
    save_player = get_player_from_save(save_file)

    if args.stockfish_location:

        if not os.path.exists(args.stockfish_location):
            print("Invalid path")
            sys.exit(1)

        save_file["stockfish_location"] = os.path.abspath(
            args.stockfish_location)

    if args.stockfish_level:
        save_file["current_game"]["stockfish_level"] = args.stockfish_level

    if args.print:
        print(board)
        end(0, save_file, config_path)

    if player != save_player:
        print("It's not your move")
        end(0, save_file, config_path)

    if not args.move:
        print("You didn't input a move")
        end(0, save_file, config_path)

    move = None

    try:
        move = board.parse_san(args.move)
        board.push(move)
    except:
        if not move:
            print("Invalid move")
            end(1, save_file, config_path)

    if board.is_checkmate():
        print("Checkmate")
        save_file["current_game"] = None
        end(0, save_file, config_path)

    if not board.is_checkmate():
        stockfish = Stockfish(
            save_file["stockfish_location"])
        stockfish.set_skill_level(
            save_file["current_game"]["stockfish_level"])

        stockfish.set_fen_position(board.fen())

        ai_move = stockfish.get_best_move()
        print(ai_move)
        board.push_san(ai_move)

        if board.is_check():
            print("Check")

        if board.is_checkmate():
            print("Checkmate")
            save_file["current_game"] = None
        else:
            save_file["current_game"]["board_fen"] = board.fen()

    write_save_file(config_path, save_file)
