import json
import os

CACHE_DIR = "cache"

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)

def get_cache(title):

    path = os.path.join(
        CACHE_DIR,
        f"{title}.json"
    )

    if os.path.exists(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return None


def save_cache(
    title,
    data
):

    path = os.path.join(
        CACHE_DIR,
        f"{title}.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )