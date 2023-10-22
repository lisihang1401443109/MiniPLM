from .trainer import PreTrainer

class ResidualPreTrainer(PreTrainer):
    def __init__(self, args, ds_config, device, do_train=True):
        super().__init__(args, ds_config, device, do_train)