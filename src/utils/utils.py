import random

def random_exclusion(start, stop, excluded) -> int:
    """Function for getting a random number with some numbers excluded"""
    excluded = set(excluded)
    value = random.randint(start, stop - len(excluded))  # Or you could use randrange
    for exclusion in tuple(excluded):
        if value < exclusion:
            break
        value += 1
    return value

def collate_fn(batch):
    return tuple(zip(*batch))