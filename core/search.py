"""
Search: Query downloaded toots by text content match and rate as +1 or -1
"""
import logging
import sqlite3
from itertools import chain

from config import API_BASE_URL
from core.utils import render_author


def existing_ratings(*, db_cursor):
    for (id_,) in db_cursor.execute(
        "SELECT id from ratings ORDER BY 75 * id + 74 % 65537"
    ):
        yield id_


def toot_search(max_single_search, *, db_cursor, search_input):
    hard_cap = max_single_search * 2
    existing = set(existing_ratings(db_cursor=db_cursor))

    count = 0

    search_matcher = search_input.lower().split()
    search_matcher = "%" + "%".join(search_matcher) + "%"

    sql_query_string = f"SELECT id, author, content, (75 * id + 74) % 65537 FROM toots WHERE LOWER(content) LIKE '{search_matcher}' ORDER BY (75 * id + 74) % 65537"

    results_count = 0
    for row in db_cursor.execute(sql_query_string):
        results_count += 1

        id_, author, content, hash_ = row

        host = API_BASE_URL

        if id_ in existing:
            continue

        count += 1
        if count > max_single_search or count > hard_cap:
            break

        print(
            "\n=== Content %3d / %3d ===\n%s\n    %s"
            % (count, max_single_search, content, render_author(author, host))
        )

        result = input("- dislike + like else skip ")

        if result in ["-", "=", "+", "0"]:
            if result == "-":
                yield (id_, author, content, -1)
            elif result == "=" or result == "+":
                print("...   + bonus!")
                max_single_search += 1
                yield (id_, author, content, 1)
            elif result == "0":
                yield (id_, author, content, 0)

        else:
            pass

    if results_count == 0:
        print("No Results Found")


def database_setup():
    # Database Setup
    with sqlite3.connect("data/db.db") as db_connection:
        db_cursor = db_connection.cursor()

        existing_tables = db_cursor.execute("SELECT name from sqlite_master")
        if "ratings" not in chain.from_iterable(existing_tables.fetchall()):
            logging.info("Creating new table ratings")
            db_cursor.execute("CREATE TABLE ratings(id, author, content, score)")


def search_impl(toots_limit, *, search_input):
    database_setup()

    with sqlite3.connect("data/db.db") as db_connection:
        db_cursor = db_connection.cursor()

        (pre_row_count,) = db_cursor.execute("select count(id) from ratings").fetchone()

        to_write = list(
            toot_search(toots_limit, db_cursor=db_cursor, search_input=search_input)
        )

        db_cursor.executemany("INSERT INTO ratings VALUES(?, ?, ?, ?)", to_write)
        db_connection.commit()

        (post_row_count,) = db_cursor.execute(
            "select count(id) from ratings"
        ).fetchone()
        logging.debug(f"Ratings added: {post_row_count} <- {pre_row_count}")

        print("\n=== Sample Positive Ratings ===")
        for idx, row in enumerate(
            db_cursor.execute(
                "SELECT id, author, content, score, (75 * id + 74) % 65537 FROM ratings where score > 0 ORDER BY (75 * id + 74) % 65537 DESC"
            )
        ):
            id_, author, content, score, shuffler = row
            print("\n", score, content)
            if idx > 3:
                break

        print("\n=== Sample Negative Ratings ===")
        for idx, row in enumerate(
            db_cursor.execute(
                "SELECT id, author, content, score, (75 * id + 74) % 65537 FROM ratings where score < 0 ORDER BY (75 * id + 74) % 65537 DESC"
            )
        ):
            id_, author, content, score, shuffler = row
            print("\n", score, content)
            if idx > 3:
                break


if __name__ == "__main__":
    search_impl(20, search_input="robotic")
