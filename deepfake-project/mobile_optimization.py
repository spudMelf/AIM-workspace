import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
#from deepfake_resnet18 import CustomResNet

example = torch.rand(1, 3, 224, 224) # doesn't matter

#model = CustomResNet()
model = torch.load('deepfake-project/models/mel_spectrogram_model_1.pt')
model = model.to('cpu')

# trace, optimize
traced_module = torch.jit.trace(model, example)
optimized_model = optimize_for_mobile(traced_module)

optimized_model.save('deepfake-project/models/mobile_mel_spectrogram_model_1.pt')