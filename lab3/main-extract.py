

def get_dataset():
    from datasets import load_dataset

    rt_trainset, rt_valset, rt_testset = load_dataset(
        "cornell-movie-review-data/rotten_tomatoes",
        split=["train", "validation", "test"])

    return rt_trainset, rt_valset, rt_testset


def main(opts):
    return None


if __name__ == "__main__":
    from types import SimpleNamespace
    from ipdb import launch_ipdb_on_exception

    config = dict()
    opts = SimpleNamespace(**config)

    with launch_ipdb_on_exception():
        main(opts)
