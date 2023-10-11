from random import randrange


def coinflip() -> bool:
    return randrange(2) == 0


def collect_negative_ids(
    n_total: int, negative_samples: int, current_id: int
) -> list[int]:
    negative_ids = []
    for _ in range(negative_samples):
        negative_id = current_id
        while (negative_id == current_id) or (negative_id in negative_ids):
            negative_id = randrange(n_total)
        negative_ids.append(negative_id)
    return negative_ids
