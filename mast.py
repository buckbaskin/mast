import argparse

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    download = subparsers.add_parser('download')
    download.add_argument('-t', '--toots', help="How many toots to download", type=int, default=1000)

    bandit = subparsers.add_parser('bandit')
    bandit.add_argument('-p', '--positive', help='Rate new toots recommended for you', action='store_true')
    bandit.add_argument('-e', '--explore', help='Rate new-to-you toots', action='store_true')
    bandit.add_argument('-n', '--negative', help='Rate new toots you might not like', action='store_true')
    bandit.add_argument('-c', '--confusion', help='Rate new toots you may or may not like', action='store_true')
    bandit.add_argument('-t', '--toots', help="How many toots to rate", type=int, default=10)

    clean = subparsers.add_parser('clean')
    clean.add_argument('-f', '--force', help="Don't ask for confirmation", action='store_true')

    report = subparsers.add_parser('report')

    args = parser.parse_args()

    print(args)
    1/0

if __name__ == '__main__':
    main()
