import torch
import logging


class ModelBase(torch.nn.Module):
    """
    base representation implementation

    """
    def __init__(self, cfg):
        super(ModelBase, self).__init__()
        self.cfg = cfg

    def forward(self, *args, **kwargs):
        raise NotImplementedError('not implemented!')

    def independent_lr_parameters(self):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        logging.info('Model {}, num of parameters: {}, set lr: {}'.format(
            self.cfg['NAME'], len(parameters), self.cfg['LEARNING_RATE']))
        if "LEARNING_RATE" in self.cfg:
            return [{
                'params': [x for x in self.parameters() if x.requires_grad],
                'lr': self.cfg['LEARNING_RATE']
            }]
        return []


class TaskBase(torch.nn.Module):
    """
    base representation implementation

    """
    def __init__(self, cfg):
        super(TaskBase, self).__init__()
        self.cfg = cfg

    def forward(self, *args, **kwargs):
        raise NotImplementedError('not implemented!')

    def get_independent_lr_parameters(self):
        parameters = []
        for model in self.model_list:
            parameters += model.independent_lr_parameters()
        return parameters
