import sqlite3
from collections import defaultdict, namedtuple
from os import walk
from os.path import join
from time import time
from typing import List

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

from stopwords import STOP_WORDS

TootContent = namedtuple(
    "TootContent", ["data", "data_titles", "target", "target_names"]
)

new_con = sqlite3.connect("data/db.db")
new_cur = new_con.cursor()

(toots_count,) = new_cur.execute("select count(id) from toots").fetchone()
(ratings_count,) = new_cur.execute("select count(id) from ratings").fetchone()

print("How it started: Ratings So Far:", ratings_count)
assert ratings_count > 0

# Nominally, find N=5 similar toots per cluster
N_CLUSTERS = toots_count // 5
MAX_NOTES = 10000


def dataset_from_db():
    data = []
    data_titles = []
    target = []
    target_names = []

    # labels = dataset.target

    for row in new_cur.execute("SELECT id, author, content, score FROM ratings"):
        if MAX_NOTES is not None and len(data) >= MAX_NOTES:
            break

        id_, author, content, score = row

        data.append(content)
        data_titles.append((id_, author))
        target.append(set([score]))
        target_names.append(score)

    rated_ids = set([d[0] for d in data_titles])

    for row in new_cur.execute("SELECT id, author, content FROM toots"):
        if MAX_NOTES is not None and len(data) >= MAX_NOTES:
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


dataset = dataset_from_db()


def vectorize_and_reduce(dataset):
    print(f"# Tags {len(dataset.target_names)}")
    print(dataset.target_names)

    print()
    print("# Dataset Summary")
    print(f"{len(dataset.data)} documents - {len(dataset.target_names)} tags")

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

    print(f"vectorization done in {time() - t0:.3f} s")

    # After ignoring terms that appear in more than 50% of the documents (as set by
    # max_df=0.5) and terms that are not present in at least 2 documents (set by
    # min_df=2), the resulting number of unique terms n_features.

    n_documents = X_tfidf.shape[0]
    print(f"n_documents: {n_documents}, n_features: {X_tfidf.shape[1]}")

    # We can additionally quantify the sparsity of the X_tfidf matrix as the
    # fraction of non-zero entries devided by the total number of elements.
    print(f"{X_tfidf.nnz / np.prod(X_tfidf.shape) * 100:05.2f} % of non-zero entries")

    # cx = X_tfidf.tocoo()
    # for idx, (doc_id, token_id, value) in enumerate(zip(cx.row, cx.col, cx.data)):
    #     if idx > 100:
    #         break
    #
    #     if value > 0.25:
    #         print(f"Doc {doc_id} Token {tokens[token_id]} : {value}")

    lsa = make_pipeline(
        TruncatedSVD(n_components=min(250, len(tokens))), Normalizer(copy=False)
    )
    t0 = time()
    X_lsa = lsa.fit_transform(X_tfidf)
    explained_variance = lsa[0].explained_variance_ratio_.sum()

    print(f"LSA done in {time() - t0:.3f} s")
    print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

    print()
    print("50 random tokens. Inspect for Stop Words")
    rng = np.random.default_rng()
    shuffle_tokens = np.copy(tokens)
    rng.shuffle(shuffle_tokens)
    print(", ".join(shuffle_tokens[:50]))

    kmeans = MiniBatchKMeans(
        n_clusters=min(N_CLUSTERS, n_documents),
        max_iter=300,
        n_init=1,
        init_size=1000,  # arbitrary
        batch_size=1000,  # arbitrary
        random_state=1969,
    )

    print("\n# Clustering")

    def fit_and_evaluate(kmeans, X):
        train_times = []
        # scores = defaultdict(list)
        t0 = time()
        kmeans.fit(X)
        train_times.append(time() - t0)
        # scores["Homogeneity"].append(metrics.homogeneity_score(labels, kmeans.labels_))
        # scores["Completeness"].append(metrics.completeness_score(labels, kmeans.labels_))
        # scores["V-measure"].append(metrics.v_measure_score(labels, kmeans.labels_))
        # scores["Adjusted Rand-Index"].append(metrics.adjusted_rand_score(labels, kmeans.labels_))
        # scores["Silhouette Coefficient"].append(
        #     metrics.silhouette_score(X, kmeans.labels_, sample_size=2000)
        # )

        print(f"clustering done in {np.mean(train_times):.2f} s ")

    fit_and_evaluate(kmeans, X_lsa)

    original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)

    return X_lsa, tokens, n_documents, original_space_centroids, kmeans


X_lsa, tokens, n_documents, original_space_centroids, kmeans = vectorize_and_reduce(
    dataset
)

order_centroids = np.argsort(original_space_centroids)[:, ::-1]

DISPLAY_COUNT = min(N_CLUSTERS, n_documents)

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


def stream_clusters():
    for idx, cluster_id in enumerate(tqdm(range(DISPLAY_COUNT))):
        if idx > 1000:
            # Essentially not taking this break code path
            break

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


def toot_cluster_rate(max_single_explore=20):
    hard_cap = max_single_explore * 2
    existing_ratings = set((i for (i,) in new_cur.execute("SELECT id FROM ratings")))

    clustered_content = sorted(
        list(
            filter(
                lambda x: x.id not in existing_ratings,
                filter(lambda x: x.cluster_positive > 0, stream_clusters()),
            )
        ),
        key=lambda x: x.cluster_positive,
        reverse=True,
    )

    count = 0

    for row in clustered_content:
        id_, author, content, confusion, positive, negative, cluster_id = row

        count += 1
        if count > max_single_explore or count > hard_cap:
            break

        print(
            "\n=== Content (%3d) %3d / %3d ===\n%s"
            % (positive, count, max_single_explore, content)
        )

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
            pass


to_write = list(toot_cluster_rate())

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
