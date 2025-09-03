# Part of this code is taken and modified from https://github.com/adap/flower
#
# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from flwr.server.server import Server as FlowerServer
from flwr.server.server import FitResultsAndFailures, fit_clients
from typing import Tuple, Optional, Dict, Union, List, Tuple
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from logging import DEBUG, INFO, WARNING
from flwr.common import Parameters, Scalar, FitIns, FitRes
from flwr.server.client_proxy import ClientProxy
import concurrent.futures

import logging
logger = logging.getLogger(__name__)


class Server(FlowerServer):
    """
    Similar to flwr.server.server.Server but pass the current round parameters
    as an argument to aggregate clients' parameters.

    Uses flower's logger.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_initial_parameters(self, timeout):
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            logger.info("Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        logger.info("Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})

        parameters_res = random_client.get_parameters(ins, timeout)
        logger.info("Received initial parameters from one random client")
        return parameters_res.parameters

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round, parameters=self.parameters, client_manager=self._client_manager
        )

        if not client_instructions:
            log(INFO, "fit_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        if len(failures) > 0:
            log(INFO, f'{len(failures)} clients failed.')
            import sys
            sys.exit(0)

        # Aggregate training results
        aggregated_result: Union[
            Tuple[Optional[Parameters], Dict[str, Scalar]],
            Optional[NDArrays],  # Deprecated
        ] = self.strategy.aggregate_fit(server_round, results, failures, current_parameters=self.parameters)

        metrics_aggregated: Dict[str, Scalar] = {}
        parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)


# def fit_clients(
#     client_instructions: List[Tuple[ClientProxy, FitIns]],
#     max_workers: Optional[int],
#     timeout: Optional[float],
# ) -> FitResultsAndFailures:
#     """Refine parameters concurrently on all selected clients."""
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         submitted_fs = {
#             executor.submit(fit_client, client_proxy, ins, timeout)
#             for client_proxy, ins in client_instructions
#         }
#         finished_fs, _ = concurrent.futures.wait(
#             fs=submitted_fs,
#             timeout=None,  # Handled in the respective communication stack
#         )

#     # Gather results
#     results: List[Tuple[ClientProxy, FitRes]] = []
#     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
#     for future in finished_fs:
#         _handle_finished_future_after_fit(
#             future=future, results=results, failures=failures
#         )
#     return results, failures


# def fit_client(
#     client: ClientProxy, ins: FitIns, timeout: Optional[float]
# ) -> Tuple[ClientProxy, FitRes]:
#     """Refine parameters on a single client."""
#     fit_res = client.fit(ins, timeout=timeout)
#     return client, fit_res
