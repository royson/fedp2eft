import ray
import torch
import threading
from typing import Optional, Type
from src.log import Checkpoint
from src.apps import App

from flwr.server.server import Server
from flwr.server.client_manager import ClientManager
from flwr.simulation.ray_transport.ray_actor import (
    VirtualClientEngineActor,
    pool_size_from_resources,
)

from src.ray_actor_custom import CustomDefaultActor, CustomVirtualClientEngineActorPool
from flwr.simulation.ray_transport.ray_client_proxy import RayActorClientProxy

import logging
logger = logging.getLogger(__name__)

def parse_ray_resources(cpus: int, vram: int):
    """ Given the amount of VRAM specified for a given experiment,
        figure out what's the corresponding ration in the GPU assigned
        for experiment. Return % of GPU to use. Here we take into account
        that the CUDA runtime allocates ~1GB upon initialization. We therefore
        substract first that amount from the total detected VRAM. CPU resources
        as returned without modification."""

    gpu_ratio = 0.0
    if torch.cuda.is_available():
        # use that figure to get a good estimate of the VRAM needed per experiment
        # (taking into account ~600MB of just CUDA init stuff)

        # Get VRAM of first GPU
        total_vram = torch.cuda.get_device_properties(0).total_memory

        # convert to MB (since the user inputs VRAM in MB)
        total_vram = float(total_vram)/(1024**2)

        # discard 1GB VRAM (which is roughtly what takes for CUDA runtime)
        # You can verify this yourself by just running:
        # `t = torch.randn(10).cuda()` (will use ~1GB VRAM)
        total_vram -= 1024

        gpu_ratio = float(vram)/total_vram
        if gpu_ratio >= 1:
            gpu_ratio = 1

        logger.info(f"GPU percentage per client: {100*gpu_ratio:.2f} % ({vram}/{total_vram})")

        # ! Limitation: this won't work well if multiple GPUs with different VRAMs are detected by Ray
        # The code above asumes therefore all GPUs have the same amount of VRAM. The same `gpu_ratio` will
        # be used in GPUs #1, #2, etc (even though there won't be 1GB taken by CUDA runtime)
        # TODO: probably we can do something smarter: run a single training batch and monitor the real memory usage. This remove user's input an no longer requiring the user to specify VRAM (which often takes a few rounds of trial-error)
    else:
        logger.warn("No CUDA device found. Disabling GPU usage for Flower clients.")

    # these keys are the ones expected by ray to specify CPU and GPU resources for each
    # Ray Task, representing a client workload.
    return {'num_cpus': cpus, 'num_gpus': gpu_ratio}


def start_simulation( 
    ckp: Checkpoint,
    server: Server,
    app: App,
) -> None:
    # Allocate client resources
    client_resources = parse_ray_resources(ckp.config.cpus, ckp.config.vram)
    logger.info(
        client_resources,
    )

    sim_config = ckp.config.simulation
    
    if ray.is_initialized():
        ray.shutdown()
        
    # Initialize Ray
    ray.init(**sim_config.ray_init_args)
    cluster_resources = ray.cluster_resources()
    logger.info(
        "Flower VCE: Ray initialized with resources: %s",
        cluster_resources,
    )
    
    actor_args = {}

    # An actor factory. This is called N times to add N actors
    # to the pool. If at some point the pool can accommodate more actors
    # this will be called again.
    def create_actor_fn() -> Type[VirtualClientEngineActor]:
        return CustomDefaultActor.options(  # type: ignore
            **client_resources,
            scheduling_strategy="DEFAULT",
        ).remote(**actor_args)

    # Instantiate ActorPool
    pool = CustomVirtualClientEngineActorPool(
        create_actor_fn=create_actor_fn,
        client_resources=client_resources,
    )

    f_stop = threading.Event()

    # Periodically, check if the cluster has grown (i.e. a new
    # node has been added). If this happens, we likely want to grow
    # the actor pool by adding more Actors to it.
    def update_resources(f_stop: threading.Event) -> None:
        """Periodically check if more actors can be added to the pool.

        If so, extend the pool.
        """
        if not f_stop.is_set():
            num_max_actors = pool_size_from_resources(client_resources)
            if num_max_actors > pool.num_actors:
                num_new = num_max_actors - pool.num_actors
                logger.info(
                    "The cluster expanded. Adding %s actors to the pool.", num_new
                )
                pool.add_actors_to_pool(num_actors=num_new)

            threading.Timer(10, update_resources, [f_stop]).start()

    update_resources(f_stop)

    # Register one RayClientProxy object for each client with the ClientManager
    for i in range(sim_config.num_clients):
        client_proxy = RayActorClientProxy(
            client_fn=app.get_client_fn(),
            cid=str(i),
            actor_pool=pool,
        )
        server.client_manager().register(client=client_proxy)

    app.run(server, timeout=8640000)
    
    server.disconnect_all_clients(timeout=100)

    if ray.is_initialized():
        ray.shutdown()
