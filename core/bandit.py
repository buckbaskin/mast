import logging
import sqlite3
from collections import defaultdict, namedtuple
from time import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans  # type: ignore
from sklearn.decomposition import TruncatedSVD  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import Normalizer  # type: ignore

from config import API_BASE_URL
from core.stopwords import STOP_WORDS
from core.utils import render_author

TootContent = namedtuple(
    "TootContent", ["data", "data_titles", "target", "target_names"]
)

ClusteredToot = namedtuple(
    "ClusteredToot",
    [
        "id",
        "author",
        "content",
        "cluster_confusion",
        "cluster_positive",
        "cluster_negative",
        "cluster_id",
    ],
)


def dataset_from_db(*, db_cursor, max_notes):
    data = []
    data_titles = []
    target = []
    target_names = []

    # labels = dataset.target

    for row in db_cursor.execute("SELECT id, author, content, score FROM ratings"):
        if max_notes is not None and len(data) >= max_notes:
            break

        id_, author, content, score = row

        data.append(content)
        data_titles.append((id_, author))
        target.append(set([score]))
        target_names.append(score)

    rated_ids = set([d[0] for d in data_titles])

    for row in db_cursor.execute("SELECT id, author, content FROM toots"):
        if max_notes is not None and len(data) >= max_notes:
            break

        id_, author, content = row

        if id_ in rated_ids:
            continue

        score = 0

        data.append(content)
        data_titles.append((id_, author))
        target.append(set([score]))
        target_names.append(score)

    target_names = sorted(list(set(target_names)))

    assert len(data) > 0
    assert len(data) == len(data_titles)
    assert len(data) == len(target)
    assert len(target_names) > 0

    dataset = TootContent(
        data=data, data_titles=data_titles, target=target, target_names=target_names
    )

    return dataset


def vectorize_and_reduce(dataset, *, n_clusters):
    logging.info("# Dataset Summary")
    logging.info(f"{len(dataset.data)} documents - {len(dataset.target_names)} tags")

    # 3 or more alphanumeric, starting with a letter.
    #   Optionally grab 't (e.g. don't) endings
    TOKEN_PATTERN = r"(?u)\b(?:3D)?[a-zA-Z]\w\w+(?:'t)?\b"

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        token_pattern=TOKEN_PATTERN,
        max_df=0.5,  # maximum in half of documents
        min_df=2,  # minimum in 2 documents
        # stop_words="english",
        stop_words=STOP_WORDS,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )

    t0 = time()
    X_tfidf = vectorizer.fit_transform(dataset.data)

    tokens = vectorizer.get_feature_names_out()

    logging.debug(f"vectorization done in {time() - t0:.3f} s")

    # After ignoring terms that appear in more than 50% of the documents (as set by
    # max_df=0.5) and terms that are not present in at least 2 documents (set by
    # min_df=2), the resulting number of unique terms n_features.

    n_documents = X_tfidf.shape[0]
    logging.debug(f"n_documents: {n_documents}, n_features: {X_tfidf.shape[1]}")

    # We can additionally quantify the sparsity of the X_tfidf matrix as the
    # fraction of non-zero entries devided by the total number of elements.
    logging.debug(
        f"{X_tfidf.nnz / np.prod(X_tfidf.shape) * 100:05.2f} % of non-zero entries"
    )

    lsa = make_pipeline(
        TruncatedSVD(n_components=min(250, len(tokens))), Normalizer(copy=False)
    )
    t0 = time()
    X_lsa = lsa.fit_transform(X_tfidf)
    explained_variance = lsa[0].explained_variance_ratio_.sum()

    logging.debug(f"LSA done in {time() - t0:.3f} s")
    logging.debug(
        f"Explained variance of the SVD step: {explained_variance * 100:.1f}%"
    )

    logging.debug("\n50 random tokens. Inspect for Stop Words")
    rng = np.random.default_rng()
    shuffle_tokens = np.copy(tokens)
    rng.shuffle(shuffle_tokens)
    logging.debug(", ".join(shuffle_tokens[:50]))

    kmeans = MiniBatchKMeans(
        n_clusters=min(n_clusters, n_documents),
        max_iter=300,
        n_init=1,
        init_size=1000,  # arbitrary
        batch_size=1000,  # arbitrary
        random_state=1969,
    )

    logging.info("\n# Clustering")

    def fit_and_evaluate(kmeans, X):
        train_times = []
        t0 = time()
        kmeans.fit(X)
        train_times.append(time() - t0)

        logging.debug(f"clustering done in {np.mean(train_times):.2f} s ")

    fit_and_evaluate(kmeans, X_lsa)

    original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)

    return X_lsa, tokens, n_documents, original_space_centroids, kmeans


def stream_clusters(*, dataset, kmeans, display_count):
    for idx, cluster_id in enumerate(range(display_count)):
        kmeans_select = kmeans.labels_ == cluster_id
        notes_in_cluster = np.count_nonzero(kmeans_select)

        if notes_in_cluster < 2:
            continue

        tags = defaultdict(int)
        for idx, label in enumerate(kmeans.labels_):
            if label == cluster_id:
                for tag in dataset.target[idx]:
                    tags[tag] += 1

        confusion = min(tags[-1], tags[1])  # larger when both are larger
        positive = tags[1]
        negative = tags[-1]

        samples_from_cluster = 0
        for idx, label in enumerate(kmeans.labels_):
            tags_for_document = dataset.target[idx]

            annotation = []
            for tag in tags_for_document:
                annotation.append(f"#{tag}")

            if label == cluster_id:
                samples_from_cluster += 1

                content = dataset.data[idx]
                title = dataset.data_titles[idx]

                id_, author = title
                yield ClusteredToot(
                    **{
                        "id": id_,
                        "author": author,
                        "content": content,
                        "cluster_confusion": confusion,
                        "cluster_positive": positive,
                        "cluster_negative": negative,
                        "cluster_id": cluster_id,
                    }
                )

                if samples_from_cluster > 5:
                    continue


def toot_cluster_rate(
    max_single_explore, sort_by, *, db_cursor, dataset, kmeans, display_count
):
    hard_cap = max_single_explore * 2
    existing_ratings = set((i for (i,) in db_cursor.execute("SELECT id FROM ratings")))

    clustered_content = filter(
        lambda x: getattr(x, f"cluster_{sort_by}") > 0 and x.id not in existing_ratings,
        stream_clusters(dataset=dataset, kmeans=kmeans, display_count=display_count),
    )

    count = 0

    for row in clustered_content:
        id_, author, content, confusion, positive, negative, cluster_id = row

        count += 1
        if count > max_single_explore or count > hard_cap:
            break

        rating = {
            "positive": positive,
            "negative": negative,
            "confusion": confusion,
        }[sort_by]

        host = API_BASE_URL
        print(
            "\n=== Content (%s %3d) %3d / %3d ===\n%s\n    %s"
            % (
                sort_by,
                rating,
                count,
                max_single_explore,
                content,
                render_author(author, host),
            )
        )

        result = input("- dislike + like else skip ")

        if result in ["-", "=", "+", "0"]:
            if result == "-":
                yield (id_, author, content, -1)
            elif result == "=" or result == "+":
                print("...   + bonus!")
                max_single_explore += 1
                yield (id_, author, content, 1)
            elif result == "0":
                yield (id_, author, content, 0)

        else:
            pass


def cluster_impl(toots_limit, sort_by):
    db_connection = sqlite3.connect("data/db.db")
    db_cursor = db_connection.cursor()

    (toots_count,) = db_cursor.execute("select count(id) from toots").fetchone()
    (ratings_count,) = db_cursor.execute("select count(id) from ratings").fetchone()

    assert ratings_count > 0

    # Nominally, find N=5 similar toots per cluster
    n_clusters = toots_count // 5
    max_notes = None

    dataset = dataset_from_db(db_cursor=db_cursor, max_notes=max_notes)

    X_lsa, tokens, n_documents, original_space_centroids, kmeans = vectorize_and_reduce(
        dataset, n_clusters=n_clusters
    )

    display_count = min(n_clusters, n_documents)

    to_write = list(
        toot_cluster_rate(
            toots_limit,
            sort_by,
            db_cursor=db_cursor,
            kmeans=kmeans,
            dataset=dataset,
            display_count=display_count,
        )
    )

    logging.debug("Prepared to write %d examples" % (len(to_write),))

    db_cursor.executemany("INSERT INTO ratings VALUES(?, ?, ?, ?)", to_write)
    db_connection.commit()

    (row_count,) = db_cursor.execute("select count(id) from ratings").fetchone()
    logging.debug(f"How it's going: Ratings So Far: {row_count}")

    print("\n=== Sample Positive Ratings ===")
    for idx, row in enumerate(
        db_cursor.execute(
            "SELECT id, author, content, score, (75 * id + 74) % 65537 FROM ratings where score > 0 ORDER BY (75 * id + 74) % 65537 DESC"
        )
    ):
        id_, author, content, score, shuffler = row
        print("\n", score, content)
        if idx > 3:
            break

    print("\n=== Sample Negative Ratings ===")
    for idx, row in enumerate(
        db_cursor.execute(
            "SELECT id, author, content, score, (75 * id + 74) % 65537 FROM ratings where score < 0 ORDER BY (75 * id + 74) % 65537 DESC"
        )
    ):
        id_, author, content, score, shuffler = row
        print("\n", score, content)
        if idx > 3:
            break


if __name__ == "__main__":
    cluster_impl(20, sort_by="positive")
