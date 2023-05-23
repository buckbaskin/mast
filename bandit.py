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

(row_count,) = new_cur.execute("select count(id) from ratings").fetchone()

print("How it started: Ratings So Far:", row_count)
assert row_count > 0


def dataset_from_directory():
    data = []
    data_titles = []
    target = []
    target_names = []

    # labels = dataset.target

    for row in new_cur.execute("SELECT id, author, content, score FROM ratings"):
        id_, author, content, score = row

        data.append(content)
        data_titles.append(tuple(id_, author))
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
