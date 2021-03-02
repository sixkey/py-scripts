from typing import Tuple
import xml.etree.ElementTree as ET
import re
import time


def parse_time(time: str) -> Tuple[int, int]:
    match = re.match(r"(\d+):(\d+)", time)
    h_s, m_s = match.groups()
    return int(h_s), int(m_s)


def remove_minutes(day: int, hour: int, minute: int,
                   delta: int) -> Tuple[int, int]:
    remove_days = 0
    remove_hours = delta // 60
    remove_minutes = delta % 60

    if remove_minutes > minute:
        remove_hours += 1
        remove_minutes = remove_minutes - minute
        minute = 60
    minute -= remove_minutes

    if remove_hours > hour:
        remove_days += 1
        remove_hours = remove_hours - hour
        hour = 24
    hour -= remove_hours

    day -= remove_days

    return day % 7, hour, minute


def tests():
    assert remove_minutes(1, 10, 15, 5) == (1, 10, 10)
    assert remove_minutes(1, 10, 5, 75) == (1, 8, 50)
    assert remove_minutes(0, 0, 0, 60) == (6, 23, 00)


if __name__ == "__main__":

    tests()

    with open("./table.xml", "r", encoding="utf-8") as f:
        tree = ET.parse(f)

        root = tree.getroot()

        lines = set()

        for day_index, item in enumerate(root.findall("tabulka/den")):
            for slot in item.findall("radek/slot"):
                od, do = slot.get("odcas"), slot.get("docas")
                kod = slot.find("akce/kod").text

                hour, minute = parse_time(od)
                day = day_index + 1
                day, hour, minute = remove_minutes(day, hour, minute, 10)

                cron = f"{str(minute)} {str(hour)} * * {day}"
                line = (f"{cron} zenity --info --text=\"{kod} at " +
                        f"{od}\"")

                if line not in lines:
                    print(line)
                lines.add(line)
