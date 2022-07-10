import os
import torch


class ModelSaver:
    '''
    save max and final checkpoint
    '''
    def __init__(self, saving_dir, saving_name):
        super().__init__()
        self.saving_dir = saving_dir
        if not os.path.exists(self.saving_dir):
            os.makedirs(saving_dir)
        self.saving_name = saving_name

        self.max_spearman = -10
        self.best_saving_path = None
        self.final_saving_path = None

    def save_model(self, model, spearman, epoch):
        model_name = self.saving_name + '_epoch{}_{:.4f}'.format(epoch, spearman)

        # if self.final_saving_path != self.best_saving_path:
        #     os.remove(self.final_saving_path)
        # self.final_saving_path = os.path.join(self.saving_dir, model_name)
        # torch.save(model, self.final_saving_path)

        # if spearman > self.max_spearman:
        #     self.max_spearman = spearman
        #     if self.best_saving_path != None:
        #         os.remove(self.best_saving_path)
        #     self.best_saving_path = self.final_saving_path

        if spearman > self.max_spearman:
            self.max_spearman = spearman
            if self.best_saving_path != None:
                os.remove(self.best_saving_path)
            self.best_saving_path = os.path.join(self.saving_dir, model_name)
            torch.save(model.state_dict(), self.best_saving_path)