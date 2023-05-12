from mastodon import Mastodon
from pprint import pprint
from bs4 import BeautifulSoup

mastodon = Mastodon(
    api_base_url="https://fosstodon.org", user_agent="formakio", ratelimit_method="wait"
)

# page = mastodon.timeline(timeline="local")

# pprint(type(page))


def cli_render(html_str, raw=False, rendered=True, links=False):
    soup = BeautifulSoup(html_str, features="html.parser")
    if raw:
        print(html_str)
    if rendered:
        print(soup.get_text())
    if links:
        collection = soup.find_all("a")
        if len(collection) > 0:
            print("Links")
            for link in collection:
                text = link.get_text()
                url = link["href"]
                print(f" - {text} -> {url}")


def toot_stream(m, *args, **kwargs):
    page = m.timeline(*args, **kwargs)

    while True:
        yield from page
        page = m.fetch_next(page._pagination_next)


for idx, toot in enumerate(toot_stream(mastodon, timeline="local")):
    if idx > 24:
        break

    print()
    print(toot["id"])
    print(" ".join(toot["account"]["acct"].split()))
    print("Content")
    cli_render(toot["content"])
