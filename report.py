"""
Report: Show authors whose content you like, ordered by the amount of their content that you like
"""
import sqlite3
from itertools import chain

# Database Setup
con = sqlite3.connect("data/db.db")
cur = con.cursor()

existing_tables = cur.execute("SELECT name from sqlite_master")
if "ratings" not in chain.from_iterable(existing_tables.fetchall()):
    raise ValueError('ratings table not in data/db.db')


new_con = sqlite3.connect("data/db.db")
new_cur = new_con.cursor()

(row_count,) = new_cur.execute("select count(id) from ratings").fetchone()

print("Number of Ratings So Far:", row_count)

(row_count,) = new_cur.execute("select count(id) from ratings where score > 0").fetchone()

print('Number of positive ratings', row_count)

####
padding = len(str(row_count))
author_padding = 20
remaining = 120 - padding - author_padding

for idx, row in enumerate(
    new_cur.execute(
        "SELECT author, count(author), max(content) FROM ratings WHERE score > 0 GROUP BY author ORDER BY count(author) DESC"
    )
):
    author, positive_cases, example_content  = row

    print(str(positive_cases).rjust(padding), author.ljust(author_padding), example_content[:remaining])

