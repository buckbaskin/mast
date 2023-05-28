"""
Report: Show authors whose content you like, ordered by the amount of their content that you like
"""
import shutil
import sqlite3
from itertools import chain

from config import API_BASE_URL
from core.utils import render_author


def database_setup():
    # Database Setup
    db_connection = sqlite3.connect("data/db.db")
    db_cursor = db_connection.cursor()

    existing_tables = db_cursor.execute("SELECT name from sqlite_master")
    if "ratings" not in chain.from_iterable(existing_tables.fetchall()):
        raise ValueError("ratings table not in data/db.db")


def report_impl(parsed_args):
    author_limit = parsed_args.count

    database_setup()

    db_connection = sqlite3.connect("data/db.db")
    db_cursor = db_connection.cursor()

    (row_count,) = db_cursor.execute("select count(id) from ratings").fetchone()

    print("Number of Ratings So Far:", row_count)

    (row_count,) = db_cursor.execute(
        "select count(id) from ratings where score > 0"
    ).fetchone()

    print("Number of positive ratings", row_count)

    ####
    padding = len(str(row_count))
    author_padding = 20
    terminal_width = shutil.get_terminal_size((100, 20)).columns
    fudge_factor = 2
    remaining = max(10, terminal_width - padding - author_padding - fudge_factor)

    host = API_BASE_URL

    for idx, row in enumerate(
        db_cursor.execute(
            "SELECT author, count(author), max(content) FROM ratings WHERE score > 0 GROUP BY author ORDER BY count(author) DESC"
        )
    ):
        if idx >= author_limit:
            break

        author, positive_cases, example_content = row

        print(
            " ".join(
                [
                    str(positive_cases).rjust(padding),
                    render_author(author, host).ljust(author_padding),
                    example_content[:remaining],
                ]
            )
        )


if __name__ == "__main__":
    report_impl(20)
