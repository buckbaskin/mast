import sqlite3
from collections import defaultdict, namedtuple
from os import walk
from os.path import join
from time import time
from typing import List

import numpy as np
from sklearn.cluster import KMeans
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
MAX_NOTES = 3200


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
        target.append(score)
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
        target.append(score)
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

    kmeans = KMeans(
        n_clusters=min(N_CLUSTERS, n_documents),
        max_iter=300,
        n_init=10,
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
