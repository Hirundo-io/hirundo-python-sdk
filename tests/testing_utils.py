import os


def get_unique_id():
    return (
        os.getenv("UNIQUE_ID", "").replace(".", "-").replace("/", "-").replace("+", "-")
    )
