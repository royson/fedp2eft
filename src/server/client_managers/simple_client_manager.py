from flwr.server.client_manager import SimpleClientManager as FlwrSimpleClientManager
from typing import Optional, List
from flwr.server.client_proxy import ClientProxy
from src.utils import get_func_from_config
import random


class SimpleClientManager(FlwrSimpleClientManager):
    '''
    Based on Flower's SimpleClientManager. 

    List of available clients is retrieved from Dataset get_available_training_clients
    '''

    def __init__(self, ckp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_config = ckp.config.data
        data_class = get_func_from_config(data_config)
        dataset = data_class(ckp, **data_config.args)

        self.available_cids = dataset.get_available_training_clients()

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        # criterion: Optional[Criterion] = None,
        # all_available: Any = False
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        available_clients = self.available_cids

        if num_clients > len(available_clients):
            return [self.clients[str(cid)] for cid in available_clients]
            # log(
            #     INFO,
            #     "Sampling failed: number of available clients"
            #     " (%s) is less than number of requested clients (%s).",
            #     len(available_clients),
            #     num_clients,
            # )
            # return []

        sampled_cids = random.sample(available_clients, num_clients)
        return [self.clients[str(cid)] for cid in sampled_cids]
