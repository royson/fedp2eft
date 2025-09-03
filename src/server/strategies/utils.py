import torch
import numpy as np 
from functools import reduce
from typing import List, Tuple, Any, Dict, cast

from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import numpy.typing as npt
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

def softmax(x, T=1.0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x/T) / np.sum(np.exp(x/T), axis=0)

def get_layer(model, layer_name):
    assert layer_name.endswith('.weight') or layer_name.endswith('.bias'), 'layer name must be learnable (end with .weight or .bias'
    layer = model
    for attrib in layer_name.split('.'):
        layer = getattr(layer, attrib)
    return layer

def aggregate_inplace(results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average. Assumes updated parameters are of the same size"""
    # Count total examples
    num_examples_total = sum([fit_res.num_examples for _, fit_res in results])

    # Compute scaling factors for each result
    scaling_factors = [
        fit_res.num_examples / num_examples_total for _, fit_res in results
    ]

    # Let's do in-place aggregation
    # get first result, then add up each other
    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
    ]
    for i, (_, fit_res) in enumerate(results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

    return params