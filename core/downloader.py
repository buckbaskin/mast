import logging
import sqlite3
from itertools import chain

from bs4 import BeautifulSoup
from mastodon import Mastodon  # type: ignore
from tqdm import tqdm

from config import API_BASE_URL


def render_paragraphs(soup):
    for p in soup.find_all("p"):
        yield p.get_text()


def cli_render(html_str, raw=False, rendered=True, links=False):
    if raw:
        yield (html_str)

    soup = BeautifulSoup(html_str, features="html.parser")
    if rendered:
        for line_break in soup.find_all("br"):
            line_break.replaceWith(" ")

        text = " ".join(render_paragraphs(soup))
        yield text

    if links:
        collection = soup.find_all("a")
        if len(collection) > 0:
            yield ("Links")
            for link in collection:
                text = link.get_text()
                url = link["href"]
                yield (f" - {text} -> {url}")


def toot_stream(m, *args, **kwargs):
    page = m.timeline(*args, **kwargs)

    while True:
        yield from page
        page = m.fetch_next(page._pagination_next)


def existing_ids_in_db():
    with sqlite3.connect("data/db.db") as db_connection:
        db_cursor = db_connection.cursor()
        return set(
            chain.from_iterable(db_cursor.execute("SELECT id from toots").fetchall())
        )


def as_db_tuples(toots_limit, *, mastodon, min_id_in_db):
    skipped = 0

    existing_ids = existing_ids_in_db()

    for idx, toot in tqdm(
        enumerate(toot_stream(mastodon, timeline="local", max_id=min_id_in_db))
    ):
        if idx >= toots_limit + skipped:
            break
        if skipped >= max(100, toots_limit * 2):
            logging.warning(f"Skipped {skipped} toots at {idx} tries. Ending early")
            break

        flattened_username = " ".join(toot["account"]["acct"].split())
        content = "\n".join(cli_render(toot["content"]))

        if toot["id"] not in existing_ids and len(content) > 0:
            yield toot["id"], flattened_username, content
        else:
            if toot["id"] in existing_ids:
                logging.debug(f"Toot {toot['id']} skipped. Skips: {skipped}")
            skipped += 1


def database_setup():
    with sqlite3.connect("data/db.db") as db_connection:
        db_cursor = db_connection.cursor()
        existing_tables = db_cursor.execute("SELECT name from sqlite_master").fetchall()
        if "toots" not in chain.from_iterable(existing_tables):
            db_cursor.execute("CREATE TABLE toots(id, author, content)")

        (min_id_in_db,) = db_cursor.execute("SELECT min(id) from toots").fetchone()
        logging.debug(f"min_id_in_db {min_id_in_db}")

        return min_id_in_db


def download_new_toots(toots_limit):
    # API Setup
    mastodon = Mastodon(
        api_base_url=API_BASE_URL,
        user_agent="mast.py",
        ratelimit_method="wait",
        ratelimit_pacefactor=0.95,
    )

    with sqlite3.connect("data/db.db") as db_connection:
        db_cursor = db_connection.cursor()
        (pre_row_count,) = db_cursor.execute("select count(id) from toots").fetchone()

        db_cursor.executemany(
            "INSERT INTO toots VALUES(?, ?, ?)",
            as_db_tuples(toots_limit, mastodon=mastodon, min_id_in_db=None),
        )
        db_connection.commit()

        (post_row_count,) = db_cursor.execute("select count(id) from toots").fetchone()

    added_count = post_row_count - pre_row_count

    return {
        "pre_row_count": pre_row_count,
        "post_row_count": post_row_count,
        "added_count": added_count,
        "remaining": toots_limit - added_count,
    }


def download_old_toots(toots_limit, min_id_in_db):
    # API Setup
    mastodon = Mastodon(
        api_base_url=API_BASE_URL,
        user_agent="mast.py",
        ratelimit_method="wait",
        ratelimit_pacefactor=0.95,
    )

    with sqlite3.connect("data/db.db") as db_connection:
        db_cursor = db_connection.cursor()
        (pre_row_count,) = db_cursor.execute("select count(id) from toots").fetchone()

        db_cursor.executemany(
            "INSERT INTO toots VALUES(?, ?, ?)",
            as_db_tuples(toots_limit, mastodon=mastodon, min_id_in_db=min_id_in_db),
        )
        db_connection.commit()

        (post_row_count,) = db_cursor.execute("select count(id) from toots").fetchone()

    added_count = post_row_count - pre_row_count

    return {
        "pre_row_count": pre_row_count,
        "post_row_count": post_row_count,
        "added_count": added_count,
        "remaining": toots_limit - added_count,
    }


def download_impl(parsed_args):
    toots_limit = parsed_args.toots

    min_id_in_db = database_setup()

    stats_new = download_new_toots(toots_limit=toots_limit)
    stats_old = download_old_toots(
        toots_limit=stats_new["remaining"], min_id_in_db=min_id_in_db
    )

    with sqlite3.connect("data/db.db") as db_connection:
        db_cursor = db_connection.cursor()

        (row_count,) = db_cursor.execute("select count(id) from toots").fetchone()

        logging.info(
            f"New: {stats_new['added_count']} Backfill: {stats_old['added_count']}"
        )
        logging.info(
            f"Toots So Far: {row_count}, Added: {row_count - stats_new['pre_row_count']}"
        )

        for idx, row in enumerate(
            db_cursor.execute("SELECT id, author, content FROM toots ORDER BY id")
        ):
            if idx > 15:
                break
            logging.info(row)


if __name__ == "__main__":
    download_impl(1000)
