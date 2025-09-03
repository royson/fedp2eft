import numpy as np
import math

def update_lr(current_round: int, total_rounds, start_lr: float, end_lr: float):
    """Applies exponential learning rate decay using the defined start_lr and end_lr.
     The basic eq is as follows:
    init_lr * exp(-round_i*gamma) = lr_at_round_i
     A more common one is :
        end_lr = start_lr*gamma^total_rounds"""

    # first we need to compute gamma, which will later be used
    # to obtain the lr for the current round

    gamma = np.power(end_lr / start_lr, 1.0 / total_rounds)
    current_lr = start_lr * np.power(gamma, current_round)
    return current_lr, gamma

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=1e-6):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr
