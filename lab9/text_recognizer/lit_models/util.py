from typing import Union

import torch


def first_element(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
    """
    Return indices of first occurence of element in x. If not found, return length of x along dim.

    Based on https://discuss.pytorch.org/t/first-nonzero-index/24769/9

    Examples
    --------
    >>> first_element(torch.tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
    tensor([2, 1, 3])
    """
    nonz = x == element
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
    ind[ind == 0] = x.shape[dim]
    return ind
