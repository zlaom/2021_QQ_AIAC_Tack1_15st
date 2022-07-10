import torch


def model_parameters_load(pretrain_path, model):
    pretrain_model = torch.load(pretrain_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrain_model.named_parameters() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model