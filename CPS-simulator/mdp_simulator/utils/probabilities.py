import numpy as np


def check_and_normalize_probabilities(probabilities: np.ndarray):
    # If the array is out of bounds just normalize it
    if probabilities.min(initial=None) < 0 or probabilities.max(initial=None) > 1:
        # Check if the min is negative:
        if probabilities.min(initial=None) < 0:
            # We need to rebase all the other probabilities with this new minimum:
            probabilities += -probabilities.min(initial=None)
        probabilities = probabilities / probabilities.sum()
        # raise Exception(f"The semantic space method relative to {self._name} didn't return a proper array "
        #                 f"[Out of Bound probability]")

    # Just make sure that the output probability is valid (close to one)
    if 0.99 < probabilities.sum() < 1.01 and probabilities.sum() != 1.0:
        # truncate to the fifth decimal - Not needed with float16
        # for i in range(probabilities.size):
        #     probabilities[i] = round(probabilities[i], 5)
        delta = 1 - probabilities.sum()
        delta = round(delta, 4)
        for i in range(probabilities.size):
            if 0 <= probabilities[i] + delta <= 1.0:
                probabilities[i] += delta
                break
        probabilities = probabilities / probabilities.sum()

    if probabilities.sum() != 1.0:
        raise ValueError("Probability out of bounds: {}".format(list(probabilities)))

    return probabilities
