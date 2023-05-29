"""
Explore: Randomly explore downloaded toots and rate as +1 or -1
"""
import logging
import sqlite3
from itertools import chain

from core.utils import label_user_input


def toot_explore(max_single_explore, *, db_cursor):
    yield from label_user_input(
        db_cursor,
        db_cursor.execute(
            "SELECT id, author, content FROM toots ORDER BY (75 * id + 74) % 65537"
        ).fetchall(),
        max_single_labelling=max_single_explore,
    )


def database_setup():
    # Database Setup
    with sqlite3.connect("data/db.db") as db_connection:
        db_cursor = db_connection.cursor()

        existing_tables = db_cursor.execute("SELECT name from sqlite_master").fetchall()
        if "ratings" not in chain.from_iterable(existing_tables):
            logging.info("Creating new table ratings")
            db_cursor.execute("CREATE TABLE ratings(id, author, content, score)")


def explore_impl(toots_limit):
    database_setup()

    with sqlite3.connect("data/db.db") as db_connection:
        db_cursor = db_connection.cursor()

        (row_count,) = db_cursor.execute("select count(id) from ratings").fetchone()

        to_write = list(toot_explore(toots_limit, db_cursor=db_cursor))

        db_cursor.executemany("INSERT INTO ratings VALUES(?, ?, ?, ?)", to_write)
        db_connection.commit()

        (row_count,) = db_cursor.execute("select count(id) from ratings").fetchone()

        print("\n=== Sample Positive Ratings ===")
        for idx, row in enumerate(
            db_cursor.execute(
                "SELECT id, author, content, score, (75 * id + 74) % 65537 FROM ratings where score > 0 ORDER BY (75 * id + 74) % 65537 DESC"
            ).fetchall()
        ):
            id_, author, content, score, shuffler = row
            print("\n", score, content)
            if idx > 3:
                break

        print("\n=== Sample Negative Ratings ===")
        for idx, row in enumerate(
            db_cursor.execute(
                "SELECT id, author, content, score, (75 * id + 74) % 65537 FROM ratings where score < 0 ORDER BY (75 * id + 74) % 65537 DESC"
            ).fetchall()
        ):
            id_, author, content, score, shuffler = row
            print("\n", score, content)
            if idx > 3:
                break


if __name__ == "__main__":
    explore_impl(20)
