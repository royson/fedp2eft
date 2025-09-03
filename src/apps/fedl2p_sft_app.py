import flwr as fl
import numpy as np
from flwr.server.server import Server
from flwr.common import Scalar
from src.apps import SFTApp
from typing import Dict, Callable, Optional, Tuple 

import logging
logger = logging.getLogger(__name__)

class FedL2PSFTApp(SFTApp):    
    def __init__(self, *args, load_fedl2p_params=None, patience=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.patience = int(patience) if patience is not None else patience
        self.pretrained_meta_weights = None
        if load_fedl2p_params is not None:
            logger.info('Loading pretrained meta-net')
            self.pretrained_meta_weights = self.ckp.offline_load(load_fedl2p_params)
            if self.pretrained_meta_weights is None:
                logger.error(f'{load_fedl2p_params} pretrained weights not found!')
                raise FileNotFoundError("pretrained weights not found")
        self.use_val_set = self.config.data.args.val_ratio > 0
        
    def get_fit_config_fn(self):
        def fit_config_fn(rnd: int) -> Dict[str, str]:
            client_config = {
                "current_round": rnd,
                }
            return client_config

        return fit_config_fn

    def get_evaluate_config_fn(self):
        """"Client evaluate. Evaluate on client's test set"""
        def evaluate_config_fn(rnd: int, **kwargs: Optional[Dict[str, Scalar]]) -> Dict[str, str]:
            client_config = {
                "current_round": rnd,
                **kwargs }
            return client_config

        return evaluate_config_fn

    def get_evaluate_fn(self) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
        """Return an evaluation function for centralized evaluation."""
        def evaluate(weights: fl.common.NDArrays, partition: str) -> Optional[Tuple[float, float]]:
            logger.warn('centralized server evaluation is disabled')
            raise NotImplementedError()
        return evaluate

    def run(self, server: Server, timeout: Optional[float]):
        """Run federated averaging for a number of rounds."""
        rnd = 0
        # Initialize parameters
        if self.pretrained_meta_weights is not None:
            server.parameters = self.pretrained_meta_weights
            logger.info('[*] Loaded pretrained meta-net weights.')        
        elif self.continue_training:
            server.parameters = self.last_run_weights
            logger.info('[*] Continuing from last run. Global Parameters Loaded.')
        else:
            server.parameters = server._get_initial_parameters(timeout=timeout)

        if self.test_only:
            logger.info("Test only - personalizing model given pretrained meta-net parameters on each client")
            server_metrics = server.evaluate_round(rnd, 
                                                   timeout=timeout, 
                                                   num_train_epochs=self.config.app.client.args.num_train_epochs,
                                                   max_steps=self.config.app.client.args.max_steps, 
                                                   seed=self.config.seed)
            logger.info(f"Personalized Result: {server_metrics}")
            self.ckp.log(server_metrics, step=rnd)
            for k, v in server_metrics.items():
                self.ckp.log_summary(k, v)
            self.ckp.save(f'results.pkl', 
                {'server_metrics': server_metrics})
            return

        # Run federated learning for num_rounds
        logger.info("FL starting")
        app_run_config = self.config.app.run

        run_log = {}
        
        lowest_loss = None
        best_server_round = self.start_run
        # best_summary = {}
        for rnd in range(self.start_run, app_run_config.num_rounds + 1):
            # Train model and replace previous global model
            server_metrics = None
            clients_metrics = None
            res_fit = server.fit_round(server_round=rnd, timeout=timeout)
            if res_fit:
                parameters_prime, _, (results, _) = res_fit  # fit_metrics_aggregated
                clients_metrics = [res[1].metrics for res in results]

                if parameters_prime:
                    server.parameters = parameters_prime
                
                if self.use_val_set:
                    mean_val_loss = np.mean([cm['val_loss'] for cm in clients_metrics])
                    if app_run_config.test_every_n is None and (lowest_loss is None or mean_val_loss < lowest_loss):
                        lowest_loss = mean_val_loss 
                        best_server_round = rnd
                        # server_metrics = server.evaluate_round(rnd, 
                        #                                         timeout=timeout, 
                        #                                         num_train_epochs=self.config.app.client.args.num_train_epochs,
                        #                                         max_steps=self.config.app.client.args.max_steps, 
                        #                                         seed=self.config.seed)
                        # logger.info(f"[Round {rnd} - Lowest Val Loss {round(lowest_loss,2)}]] {server_metrics}")
                        # best_summary = server_metrics
                        logger.info(f"[Round {rnd} - Lowest Val Loss {round(lowest_loss,2)}]]")

                        # self.ckp.log(server_metrics, step=rnd)
                        self.ckp.offline_save(f'{self.config.model_directory}/{self.ckp.name}/best_weights.pkl',
                            server.parameters)

                if app_run_config.test_every_n is not None and rnd % app_run_config.test_every_n == 0:
                    logger.debug(f"[Round {rnd}] Evaluating meta-net pretrained weights on test set.")
                    server_metrics = server.evaluate_round(rnd, 
                                                        timeout=timeout, 
                                                        num_train_epochs=self.config.app.client.args.num_train_epochs,
                                                        max_steps=self.config.app.client.args.max_steps, 
                                                        seed=self.config.seed)
                    self.ckp.log(server_metrics, step=rnd)
                    logger.info(f"[Round {rnd}] {server_metrics}")

            # end of round saving
            run_log[rnd] = {'clients_metrics': clients_metrics, 
                'server_metrics': server_metrics}
            self.ckp.save('results.pkl', run_log)            
            self.ckp.offline_save(f'{self.config.model_directory}/{self.ckp.name}/latest_weights.pkl',
                server.parameters)
            if rnd > 0 and app_run_config.save_every_n is not None and rnd % app_run_config.save_every_n == 0:
                self.ckp.offline_save(f'{self.config.model_directory}/{self.ckp.name}/weights_round_{rnd}.pkl',
                    server.parameters)
            self.ckp.save(f'last_round_saved.pkl', rnd)

            # check patience
            if self.patience is not None and rnd - best_server_round >= self.patience:
                logger.info(f'Round {rnd} exceed patience value of {self.patience}. Ending training.')
                break            

        self.ckp.offline_save(f'{self.config.model_directory}/{self.ckp.name}/weights_{rnd}_final.pkl',
                server.parameters)
        
        # if best_summary:
        #     for k,v in best_summary.items():
        #         logger.info(f'Logging {k}:{v}')
        #         self.ckp.log_summary(k, v)        
