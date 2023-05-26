import argparse
from functools import partial

from explore import explore_impl
from bandit import cluster_impl
from report import report_impl
from downloader import download_impl


def bandit_impl(parsed_args):
    user_selection = list(
        filter(
            lambda t: t[1],
            (
                (k, getattr(parsed_args, k))
                for k in ["positive", "explore", "confusion", "negative"]
            ),
        )
    )

    if len(user_selection) > 1:
        raise ValueError(
            "Please select one option from: %s" % ([k for k, _ in user_selection],)
        )

    if len(user_selection) == 0:
        user_selection = "positive"
    else:
        user_selection = user_selection[0][0]

    print("user_selection", user_selection)

    {
        "confusion": partial(cluster_impl, sort_by="confusion"),
        "explore": explore_impl,
        "negative": partial(cluster_impl, sort_by="negative"),
        "positive": partial(cluster_impl, sort_by="positive"),
    }[user_selection](parsed_args.toots)


def clean_impl(parsed_args):
    raise NotImplementedError(
        "clean not implemented. Please manually delete the .db file"
    )


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    download = subparsers.add_parser("download")
    download.add_argument(
        "-t", "--toots", help="How many toots to download", type=int, default=1000
    )
    download.set_defaults(func=download_impl)

    bandit = subparsers.add_parser("bandit")
    bandit.add_argument(
        "-p",
        "--positive",
        help="Rate new toots recommended for you",
        action="store_true",
    )
    bandit.add_argument(
        "-e", "--explore", help="Rate new-to-you toots", action="store_true"
    )
    bandit.add_argument(
        "-n",
        "--negative",
        help="Rate new toots you might not like",
        action="store_true",
    )
    bandit.add_argument(
        "-c",
        "--confusion",
        help="Rate new toots you may or may not like",
        action="store_true",
    )
    bandit.add_argument(
        "-t", "--toots", help="How many toots to rate", type=int, default=10
    )
    bandit.set_defaults(func=bandit_impl)

    clean = subparsers.add_parser("clean")
    clean.add_argument(
        "-f", "--force", help="Don't ask for confirmation", action="store_true"
    )
    clean.set_defaults(func=clean_impl)

    report = subparsers.add_parser("report")
    report.add_argument('-n', '--count', help='Number of authors to report', type=int, default=20)
    report.set_defaults(func=report_impl)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
