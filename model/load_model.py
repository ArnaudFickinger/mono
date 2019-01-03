import torch
from .hourglass_model import hourglass

class ModelLoader():
    def name(self):
        return 'CleanModel'

    def __init__(self):
        model = hourglass
        model= torch.nn.parallel.DataParallel(model, device_ids = [0])
        model_parameters = torch.load("./model/best_param.pth")
        model.load_state_dict(model_parameters)
        if torch.cuda.is_available():
            self.netG = model.cuda()
        else:
            self.netG = model

    def switch_to_train(self):
        self.netG.train()

    def switch_to_eval(self):
        self.netG.eval()
    
