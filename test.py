import torch
from layers.Attention.MutliHeadAttention import MultiheadAttention

from utils.Data.HARDataset import getData

#################################################### setup ###########################################
print("Using torch", torch.__version__)
torch.manual_seed(42)
# device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################################## testing ##############################################

classes = ('Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing')
batch_size = 10
def evalModel(model, data_loader):
  model.eval()
  with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(6)]
    n_class_samples = [0 for i in range(6)]
    for sequence, labels in data_loader:
        sequence = sequence.to(device)
        labels = labels.to(device)
        outputs = model(sequence)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)

        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %\n')

    for i in range(6):
        if n_class_samples[i]>0:
          acc = 100.0 * n_class_correct[i] / n_class_samples[i]
          print(f'Accuracy of {classes[i]}: {acc} %')
        else:
          print(f'Accuracy of {classes[i]}: {0} %')

def getModelSize(model):
  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print('model size: {:.3f}MB'.format(size_all_mb))

if __name__ == '__main__':
   # getting dataloaders
    train_data_loader, test_data_loader = getData()
   
   # load the saved model
    dir = 'C:/Users/Admin/Desktop/CNN_HAR/Code/utils/SavedModels/'
    state_dict = torch.load(dir + 'mainmodel.tar')

    # Create a new model and load the state
    model = MultiheadAttention(3000)
    model.load_state_dict(state_dict)

    # testing model
    evalModel(model, test_data_loader)

    # testing Quantized Model
    q_state_dict = torch.load(dir + 'q_model.tar')
    q_model = MultiheadAttention(3000)
    model.load_state_dict(q_state_dict)

    # tesing Quantized Model
    evalModel(q_model, test_data_loader)

    # comparing amount of Quantization
    q_modelSize = getModelSize(q_model)
    main_modelSize = getModelSize(model)

    print(q_modelSize, main_modelSize)