import os
from mastodon import Mastodon
from pprint import pprint
from bs4 import BeautifulSoup
from tqdm import tqdm

mastodon = Mastodon(
    api_base_url="https://fosstodon.org", user_agent="formakio", ratelimit_method="wait"
)

# page = mastodon.timeline(timeline="local")

# pprint(type(page))


def cli_render(html_str, raw=False, rendered=True, links=False):
    soup = BeautifulSoup(html_str, features="html.parser")
    if raw:
        yield (html_str)
    if rendered:
        yield (soup.get_text())
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


for idx, toot in tqdm(enumerate(toot_stream(mastodon, timeline="local"))):
    if idx >= 10000:
        break

    id_ = str(toot["id"])
    try:
        open(os.path.join("data", id_))
    except FileNotFoundError:
        with open(os.path.join("data", id_), "w") as f:
            flattened_username = " ".join(toot["account"]["acct"].split())
            f.write(
                "\n".join(
                    [
                        str(id_),
                        flattened_username,
                        "Content",
                        "\n".join(cli_render(toot["content"])),
                    ]
                )
            )
