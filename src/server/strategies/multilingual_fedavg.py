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
import numpy as np
from src.server.strategies import FedAvg
from src.server.strategies.utils import aggregate_inplace
from flwr.server.client_manager import ClientManager
from collections import defaultdict

from typing import Dict, Optional, Tuple, List
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
    FitRes,
    ndarrays_to_parameters,
)

import logging
logger = logging.getLogger(__name__)


class MultilingualFedAvg(FedAvg):
    def __init__(self, *args, task='generation', **kwargs):
        super().__init__(*args, **kwargs)
        assert task in ['generation', 'classification']
        self.task = task

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        aggregated_weights = aggregate_inplace(results)

        # log training loss
        train_summary = defaultdict(list)
        for _, fit_res in results:
            for m, v in fit_res.metrics.items():
                train_summary[m].append(v)
        for k, v in train_summary.items():
            self.ckp.log({f'mean_{k}': np.mean(v)},
                         step=server_round, commit=False)
            self.ckp.log({f'var_{k}': np.var(v)},
                         step=server_round, commit=False)

        return ndarrays_to_parameters(aggregated_weights), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager, **kwargs: Optional[Dict[str, Scalar]],
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round, **kwargs)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        # if server_round >= 0:
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # else:
        #     clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Dict[str, Scalar]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        client_results = {}
        for client, evaluate_res in results:
            client_results[client.cid] = evaluate_res.metrics

        if self.task == 'generation':
            aggregated_metrics = compute_multilingual_metric(
                client_results
            )
        else:
            aggregated_metrics = compute_multilingual_classification_metric(
                client_results
            )

        return aggregated_metrics


def compute_multilingual_classification_metric(results):
    """Aggregate evaluation results obtained from multiple clients."""
    aggregated_metrics = defaultdict(float)
    metrics = {}
    mean_train_loss = []
    languages = set()
    # all_accuracies = []
    for _, client_metrics in results.items():
        languages.add(client_metrics['language'])
        aggregated_metrics[f"{client_metrics['language']}_correct"] += client_metrics['accuracy'] * \
            client_metrics['count']
        aggregated_metrics[f"{client_metrics['language']}_total"] += client_metrics['count']
        # aggregated_metrics[f"{client_metrics['language']}_accuracy"] = client_metrics['accuracy']
        # all_accuracies.append(client_metrics['accuracy'])
        if 'train_loss' in client_metrics:
            mean_train_loss.append(client_metrics['train_loss'])

    all_accuracies = []
    for language in languages:
        metrics[f'{language}_accuracy'] = aggregated_metrics[f"{language}_correct"] / \
            aggregated_metrics[f"{language}_total"] * 100
        all_accuracies.append(metrics[f'{language}_accuracy'])

    metrics['avg_accuracy'] = np.mean(all_accuracies)
    if mean_train_loss:
        metrics['mean_train_loss'] = np.mean(mean_train_loss)

    return metrics


def compute_multilingual_metric(results):
    """Aggregate evaluation results obtained from multiple clients."""
    eval_metrics = ['rouge1', 'rouge2', 'rougeL', 'meteor']

    aggregated_metrics = defaultdict(float)
    languages = set()
    # aggregating all the scores and num of samples for each language
    for _, client_metrics in results.items():
        for language in client_metrics['languages']:
            languages.add(language)

            aggregated_metrics[f'{language}_count'] += client_metrics[f'{language}_count']
            for eval_metric in eval_metrics:
                aggregated_metrics[f'{language}_{eval_metric}'] += client_metrics[f'{language}_{eval_metric}']

    # total score / num of samples for each language
    for eval_metric in eval_metrics:
        avg_score = 0.
        for language in languages:
            aggregated_metrics[f'{language}_{eval_metric}'] /= aggregated_metrics[f'{language}_count']
            avg_score += aggregated_metrics[f'{language}_{eval_metric}']
        avg_score /= len(languages)
        aggregated_metrics[f'avg_{eval_metric}'] = avg_score

    # remove all count
    for k in list(aggregated_metrics.keys()):
        if '_count' in k:
            del aggregated_metrics[k]

    return aggregated_metrics
