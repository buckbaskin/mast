"""
Requires user authentication
"""
import sqlite3
from itertools import chain

from bs4 import BeautifulSoup
from mastodon import Mastodon  # type: ignore
from tqdm import tqdm

from config import API_BASE_URL

# Database Setup
con = sqlite3.connect("data/db.db")
cur = con.cursor()
existing_tables = cur.execute("SELECT name from sqlite_master")
if "ratings" not in chain.from_iterable(existing_tables.fetchall()):
    cur.execute("CREATE TABLE ratings(id, author, content, score)")

(max_id_in_db,) = cur.execute("SELECT max(id) from ratings").fetchone()
print("max_id_in_db", max_id_in_db)

# API Setup
mastodon = Mastodon(
    api_base_url=API_BASE_URL, user_agent="formakio", ratelimit_method="wait"
)


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


def favorites_stream(m, *args, **kwargs):
    page = m.favourites(*args, **kwargs)

    while True:
        yield from page
        page = m.fetch_next(page._pagination_next)


def as_db_tuples():
    for idx, toot in tqdm(enumerate(favorites_stream(mastodon, min_id=max_id_in_db))):
        if idx >= 10000:
            break

        id_ = str(toot["id"])
        flattened_username = " ".join(toot["account"]["acct"].split())
        content = "\n".join(cli_render(toot["content"]))

        if len(content) > 0:
            yield toot["id"], flattened_username, content, 1


try:
    cur.executemany("INSERT INTO ratings VALUES(?, ?, ?, ?)", as_db_tuples())
    con.commit()
finally:
    con.close()
    del cur
    del con

new_con = sqlite3.connect("data/db.db")
new_cur = new_con.cursor()

(row_count,) = new_cur.execute("select count(id) from ratings").fetchone()

print("Ratings So Far:", row_count)

for idx, row in enumerate(
    new_cur.execute("SELECT id, author, content, score FROM ratings ORDER BY id")
):
    if idx > 15:
        break
    print(row)
