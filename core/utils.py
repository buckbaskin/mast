import logging

from config import API_BASE_URL


def render_author(author, host):
    if host.startswith("http://"):
        host = host[len("http://") :]
    if host.startswith("https://"):
        host = host[len("https://") :]
    return f"@{author}@{host}"


def existing_ratings(db_cursor):
    def existing_ratings_impl(db_cursor):
        for (id_,) in db_cursor.execute(
            "SELECT id from ratings ORDER BY 75 * id + 74 % 65537"
        ).fetchall():
            yield id_

    # Don't stream these ids, instead return a set. Otherwise, the cursor can silently mix results from different queries :(
    return set(existing_ratings_impl(db_cursor))


def label_user_input(db_cursor, unlabeled_toot_stream, *, max_single_search):
    hard_cap = max_single_search * 2
    existing_rating_ids = existing_ratings(db_cursor)

    skipped = 0

    error_rows = []

    empty_results = True

    for idx, row in enumerate(unlabeled_toot_stream):
        empty_results = False

        try:
            id_, author, content = row
        except ValueError:
            logging.debug(f"row {type(row)} {[type(i) for i in row]} {row}")
            error_rows.append(row)
            skipped += 1
            continue

        host = API_BASE_URL

        if id_ in existing_rating_ids:
            skipped += 1
            continue

        if idx > max_single_search or idx > hard_cap:
            break

        print(
            "\n=== Content %3d / %3d ===\n%s\n    %s"
            % (idx, max_single_search, content, render_author(author, host))
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

    if len(error_rows) > 0:
        logging.warn("Some results returned an error before they could be rated")

    if skipped > 0:
        print("Already Rated Some Results")
    elif empty_results:
        print("No Results Found")
