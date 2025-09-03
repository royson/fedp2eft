import flwr as fl
from flwr.server.server import Server
from flwr.common import Scalar
import torch
import gc
from src.apps import App
from src.utils import get_func_from_config
from src.apps.app_utils import cosine_learning_rate
from typing import Dict, Callable, Optional, Tuple 
from flwr.common import parameters_to_ndarrays
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from collections import OrderedDict
from src.data import Meteor

import logging
logger = logging.getLogger(__name__)

class SFTApp(App):    
    def __init__(self, *args, test_only=False, skip_initial_eval=True, **kwargs):
        super().__init__(*args, **kwargs)
        last_round = self.ckp.load('last_round_saved.pkl') # continual training
        self.config = self.ckp.config
        self.start_run = 1
        self.test_only = test_only
        self.skip_initial_eval = skip_initial_eval
 
        # resume training
        self.continue_training = False
        if last_round is not None:
            self.last_run_weights = self.ckp.offline_load(f'{self.config.model_directory}/{self.ckp.name}/latest_weights.pkl')
            self.start_run = last_round + 1
            self.continue_training = True
            logger.info(f'Starting from round {self.start_run}..')

        ### Check if model and tokenizer has been downloaded, if not download it
        tokenizer_config = self.config.models.tokenizer
        arch_fn = get_func_from_config(tokenizer_config)
        tokenizer = arch_fn(**tokenizer_config.args)
        net_config = self.config.models.net
        arch_fn = get_func_from_config(net_config)
        net = arch_fn(**net_config.args, tokenizer=tokenizer)

        ### check if evaluation dataset is downloaded
        meteor_metric = Meteor()
        meteor_metric._download_and_prepare(None) 

        if hasattr(net.config, 'quantization_config'):
            net = net.dequantize()
        net.cpu()
        del net
        del tokenizer
        del meteor_metric
        gc.collect()
        torch.cuda.empty_cache()
        ### 
        
    def get_fit_config_fn(self):
        """Return a configuration with static batch size and (local) epochs."""
        def fit_config_fn(rnd: int) -> Dict[str, str]:
            fit_config = self.config.app.on_fit

            if fit_config.mode == 'constant':
                lr = fit_config.lr
            elif fit_config.mode == 'cosine':
                lr = cosine_learning_rate(rnd, self.config.app.run.num_rounds, initial_lr=fit_config.lr)
            else:
                raise NotImplementedError()

            self.ckp.log({"global_LR": lr}, step=rnd)

            client_config = {
                "lr": lr,
                "current_round": rnd,
                }
            return client_config

        return fit_config_fn

    def get_evaluate_config_fn(self):
        """"Client evaluate. Evaluate on client's test set"""
        def evaluate_config_fn(rnd: int, **kwargs: Optional[Dict[str, Scalar]]) -> Dict[str, str]:
            eval_config = self.config.app.on_evaluate

            client_config = {
                "lr": eval_config.lr,
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
        if self.continue_training:
            server.parameters = self.last_run_weights
            logger.info('[*] Continuing from last run. Global Parameters Loaded.')
        else:
            server.parameters = server._get_initial_parameters(timeout=timeout)

            # Get initial test accuracy - for debugging
            if not self.skip_initial_eval:
                server_metrics = server.evaluate_round(0, 
                                                    timeout=timeout, 
                                                    num_train_epochs=0, 
                                                    max_steps=0, 
                                                    seed=self.config.seed)
                self.ckp.log(server_metrics, step=rnd)
                for k,v in server_metrics.items():
                    logger.info(f'{k}:{v}')

        if self.test_only:
            logger.info("Test only - personalizing pretrained model on each client")
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

            if rnd % app_run_config.test_every_n == 0:
                logger.debug(f"[Round {rnd}] Evaluating global model on test set.")
                server_metrics = server.evaluate_round(rnd, 
                                                       timeout=timeout, 
                                                       num_train_epochs=0,
                                                       max_steps=0, 
                                                       seed=self.config.seed)
                self.ckp.log(server_metrics, step=rnd)
                logger.info(f"[Round {rnd}] {server_metrics}")

            # end of round saving
            run_log[rnd] = {'clients_metrics': clients_metrics, 
                'server_metrics': server_metrics}
            self.ckp.save('results.pkl', run_log)            
            self.ckp.offline_save(f'{self.config.model_directory}/{self.ckp.name}/latest_weights.pkl',
                server.parameters)
            if rnd > 0 and rnd % app_run_config.save_every_n == 0:
                self.ckp.offline_save(f'{self.config.model_directory}/{self.ckp.name}/weights_round_{rnd}.pkl',
                    server.parameters)
            self.ckp.save(f'last_round_saved.pkl', rnd) # for continual training

        self.ckp.offline_save(f'{self.config.model_directory}/{self.ckp.name}/weights_{rnd}_final.pkl',
                server.parameters)
        
        # merge and save pretrained model without quantization
        net_config = self.config.models.net
        arch_fn = get_func_from_config(net_config)
        net_config_without_quant = {**net_config.args, 'quantization': None}
        net = arch_fn(**net_config_without_quant)
        weights = parameters_to_ndarrays(server.parameters)

        peft_state_dict_keys = get_peft_model_state_dict(net, save_embedding_layers=False).keys()
        params_dict = zip(peft_state_dict_keys, weights)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(net, state_dict)
        net = net.merge_and_unload()
        net.save_pretrained(f'{self.config.model_directory}/{self.ckp.name}/model.pt')
