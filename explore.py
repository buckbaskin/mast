"""
Explore: Randomly explore downloaded toots and rate as +1 or -1
"""
import sqlite3
from itertools import chain

# Database Setup
con = sqlite3.connect("data/db.db")
cur = con.cursor()

existing_tables = cur.execute("SELECT name from sqlite_master")
if "ratings" not in chain.from_iterable(existing_tables.fetchall()):
    print("Creating new table ratings")
    cur.execute("CREATE TABLE ratings(id, author, content, score)")


new_con = sqlite3.connect("data/db.db")
new_cur = new_con.cursor()

(row_count,) = new_cur.execute("select count(id) from ratings").fetchone()

print("How it started: Ratings So Far:", row_count)

print("Random toots:")


def existing_ratings():
    for (id_,) in new_cur.execute(
        "SELECT id from ratings ORDER BY 75 * id + 74 % 65537"
    ):
        yield id_


def toot_explore(max_single_explore = 20):
    hard_cap = max_single_explore * 2
    existing = set(existing_ratings())

    count = 0

    for idx, row in enumerate(
        new_cur.execute(
            "SELECT id, author, content, (75 * id + 74) % 65537 FROM toots ORDER BY (75 * id + 74) % 65537"
        )
    ):
        id_, author, content, hash_ = row

        if id_ in existing:
            continue

        count += 1
        if count > max_single_explore or count > hard_cap:
            break

        print("\n=== Content %3d / %3d ===\n%s" % (count, max_single_explore, content))

        result = input("- dislike + like else skip ")

        if result in ["-", "=", "+", "0"]:
            if result == "-":
                # print("-")
                yield (id_, author, content, -1)
            elif result == "=" or result == "+":
                print("...   + bonus!")
                max_single_explore += 1
                yield (id_, author, content, 1)
            elif result == "0":
                # print("0")
                yield (id_, author, content, 0)

        else:
            # print("skipped")
            pass


to_write = list(toot_explore())

print("Prepared to write %d examples" % (len(to_write),))

new_cur.executemany("INSERT INTO ratings VALUES(?, ?, ?, ?)", to_write)
new_con.commit()

(row_count,) = new_cur.execute("select count(id) from ratings").fetchone()
print("How it's going: Ratings So Far:", row_count)

print("\n=== Sample Positive Ratings ===")
for idx, row in enumerate(
    new_cur.execute(
        "SELECT id, author, content, score, (75 * id + 74) % 65537 FROM ratings where score > 0 ORDER BY (75 * id + 74) % 65537 DESC"
    )
):
    id_, author, content, score, shuffler = row
    print("\n", score, content)
    if idx > 3:
        break

print("\n=== Sample Negative Ratings ===")
for idx, row in enumerate(
    new_cur.execute(
        "SELECT id, author, content, score, (75 * id + 74) % 65537 FROM ratings where score < 0 ORDER BY (75 * id + 74) % 65537 DESC"
    )
):
    id_, author, content, score, shuffler = row
    print("\n", score, content)
    if idx > 3:
        break
