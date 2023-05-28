import argparse
import logging
from functools import partial

from core.bandit import cluster_impl
from core.downloader import download_impl
from core.explore import explore_impl
from core.report import report_impl


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
    download.add_argument(
        "-v", "--verbosity", action="count", help="Increase Logging Level"
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
    bandit.add_argument(
        "-v", "--verbosity", action="count", help="Increase Logging Level"
    )
    bandit.set_defaults(func=bandit_impl)

    clean = subparsers.add_parser("clean")
    clean.add_argument(
        "-f", "--force", help="Don't ask for confirmation", action="store_true"
    )
    clean.add_argument(
        "-v", "--verbosity", action="count", help="Increase Logging Level"
    )
    clean.set_defaults(func=clean_impl)

    report = subparsers.add_parser("report")
    report.add_argument(
        "-n", "--count", help="Number of authors to report", type=int, default=20
    )
    report.add_argument(
        "-v", "--verbosity", action="count", help="Increase Logging Level"
    )
    report.set_defaults(func=report_impl)

    args = parser.parse_args()
    if args.verbosity is None:
        logging.basicConfig(level=logging.WARNING)
    elif args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbosity >= 2:
        logging.basicConfig(level=logging.DEBUG)
    args.func(args)


if __name__ == "__main__":
    main()
