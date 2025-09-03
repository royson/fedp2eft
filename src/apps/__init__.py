from .app import App
from .sft_app import SFTApp
from .fedl2p_sft_app import FedL2PSFTApp
from .seqcls_app import SeqClsApp
from .fedl2p_seqcls_app import FedL2PSeqClsApp

from src.utils import get_func_from_config

import logging
logger = logging.getLogger(__name__)


def get_app(ckp):
    app_config = ckp.config.app
    app_fn = get_func_from_config(app_config)

    return app_fn(ckp, **app_config.args)
