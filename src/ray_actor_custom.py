# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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

import ray
import threading
from flwr.simulation.ray_transport.ray_actor import (
    VirtualClientEngineActor,
    VirtualClientEngineActorPool,
    pool_size_from_resources)
from ray.util.actor_pool import ActorPool
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union


@ray.remote
class CustomDefaultActor(VirtualClientEngineActor):
    """A Custom version of Ray Actor class that runs client runs.

    Parameters
    ----------
    on_actor_init_fn: Optional[Callable[[], None]] (default: None)
        A function to execute upon actor initialization.
    """

    def __init__(self, on_actor_init_fn: Optional[Callable[[], None]] = None) -> None:
        super().__init__()
        if on_actor_init_fn:
            on_actor_init_fn()

    def ready(self):
        pass


class CustomVirtualClientEngineActorPool(VirtualClientEngineActorPool):
    """A Custom version of VirtualClientEngineActorPool from flwr

    Parameters
    ----------
    create_actor_fn : Callable[[], Type[VirtualClientEngineActor]]
        A function that returns an actor that can be added to the pool.

    client_resources : Dict[str, Union[int, float]]
        A dictionary specifying the system resources that each
        actor should have access. This will be used to calculate
        the number of actors that fit in your cluster. Supported keys
        are `num_cpus` and `num_gpus`. E.g. {`num_cpus`: 2, `num_gpus`: 0.5}
        would allocate two Actors per GPU in your system assuming you have
        enough CPUs. To understand the GPU utilization caused by `num_gpus`,
        as well as using custom resources, please consult the Ray documentation.

    actor_lists: List[VirtualClientEngineActor] (default: None)
        This argument should not be used. It's only needed for serialization purposes
        (see the `__reduce__` method). Each time it is executed, we want to retain
        the same list of actors.
    """

    def __init__(
        self,
        create_actor_fn: Callable[[], Type[VirtualClientEngineActor]],
        client_resources: Dict[str, Union[int, float]],
        actor_list: Optional[List[Type[VirtualClientEngineActor]]] = None,
    ):
        self.client_resources = client_resources
        self.create_actor_fn = create_actor_fn

        if actor_list is None:
            # Figure out how many actors can be created given the cluster resources
            # and the resources the user indicates each VirtualClient will need
            num_actors = pool_size_from_resources(client_resources)
            actors = [create_actor_fn() for _ in range(num_actors)]
        else:
            # When __reduce__ is executed, we don't want to created
            # a new list of actors again.
            actors = actor_list

        # import time
        # stime = time.time()
        ray.get([a.ready.remote() for a in actors])
        # print("Ready really takes:", time.time() - stime)
        super(VirtualClientEngineActorPool, self).__init__(actors)

        # A dict that maps cid to another dict containing: a reference to the remote job
        # and its status (i.e. whether it is ready or not)
        self._cid_to_future: Dict[
            str, Dict[str, Union[bool, Optional[ObjectRef[Any]]]]
        ] = {}
        self.actor_to_remove: Set[str] = set()  # a set
        self.num_actors = len(actors)

        self.lock = threading.RLock()
