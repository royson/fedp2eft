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
from flwr.server.server import EvaluateResultsAndFailures, FitResultsAndFailures, evaluate_clients, fit_clients
from typing import Tuple, Optional, Dict, Union, List, Tuple
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.common import (
    Parameters,
    Scalar,
)
from logging import DEBUG, INFO, WARNING

import logging
logger = logging.getLogger(__name__)


class LLMServer(FlowerServer):
    """
    Similar to flwr.server.server.Server but pass extra keyword arguments to evaluate

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

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
        **kwargs: Optional[Dict[str, Scalar]],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
            **kwargs,  # additional kwargs
        )
        if not client_instructions:
            log(INFO, "configure_evaluate: no clients selected, skipping evaluation")
            return None
        log(
            INFO,
            "configure_evaluate: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(results),
            len(failures),
        )

        if len(failures) > 0:
            log(INFO, f'{len(failures)} clients failed.')
            import os
            os._exit(0)

        # Aggregate the evaluation results
        return self.strategy.aggregate_evaluate(server_round, results, failures)

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
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        if len(failures) > 0:
            log(INFO, f'{len(failures)} clients failed.')
            import os
            os._exit(0)

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)
