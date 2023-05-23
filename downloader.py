from mastodon import Mastodon
from bs4 import BeautifulSoup
from tqdm import tqdm
import sqlite3
from itertools import chain

# Database Setup
con = sqlite3.connect("data/db.db")
cur = con.cursor()
existing_tables = cur.execute("SELECT name from sqlite_master")
if "toots" not in chain.from_iterable(existing_tables.fetchall()):
    cur.execute("CREATE TABLE toots(id, author, content)")

(min_id_in_db,) = cur.execute("SELECT min(id) from toots").fetchone()
print(min_id_in_db)

# API Setup
mastodon = Mastodon(
    api_base_url="https://fosstodon.org",
    user_agent="formakio",
    ratelimit_method="wait",
    ratelimit_pacefactor=0.95,
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


def toot_stream(m, *args, **kwargs):
    page = m.timeline(*args, **kwargs)

    while True:
        yield from page
        page = m.fetch_next(page._pagination_next)


def as_db_tuples():
    for idx, toot in tqdm(
        enumerate(toot_stream(mastodon, timeline="local", max_id=min_id_in_db))
    ):
        if idx >= 1000:
            break

        flattened_username = " ".join(toot["account"]["acct"].split())
        content = "\n".join(cli_render(toot["content"]))

        if len(content) > 0:
            yield toot["id"], flattened_username, content


try:
    cur.executemany("INSERT INTO toots VALUES(?, ?, ?)", as_db_tuples())
    con.commit()
finally:
    con.close()
    del cur
    del con

new_con = sqlite3.connect("data/db.db")
new_cur = new_con.cursor()

(row_count,) = new_cur.execute("select count(id) from toots").fetchone()

print("Toots So Far:", row_count)

for idx, row in enumerate(
    new_cur.execute("SELECT id, author, content FROM toots ORDER BY id")
):
    if idx > 15:
        break
    print(row)
