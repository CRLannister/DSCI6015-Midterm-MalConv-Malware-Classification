import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
import ember

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

np.int = np.int64

class MalConv(nn.Module):
    def __init__(self, input_length=3000000, embedding_dim=8, window_size=5, output_dim=1):
        super(MalConv, self).__init__()
        self.embed = nn.Embedding(input_length, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=window_size, stride=window_size - 3, padding=0)
        self.conv2 = nn.Conv1d(embedding_dim, 64, kernel_size=window_size, stride=window_size - 3, padding=0)
        self.gating = nn.Sigmoid()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x)
        # Convert to (batch_size, channels, length)
        x = x.transpose(1, 2)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        # Element-wise multiplication
        gated = conv1 * self.gating(conv2)
         # Remove the last dimension
        global_max_pool = self.global_max_pool(gated).squeeze(2)
        fc1 = F.relu(self.fc1(global_max_pool))
        fc2 = self.fc2(fc1)
        output = self.sigmoid(fc2)
        return output


def feature_vector(pe_data):
    extract = ember.PEFeatureExtractor() 
    data = extract.feature_vector(pe_data) #vectorizing the extracted features
    # data = ember.read_vectorized_features(pe_data)
    MM = None
    scaler_filename = '/gdrive/MyDrive/vMalConv/minmax_scaler.pkl'
    with open(scaler_filename, "rb") as f:
        MM = pickle.load(f)

    # print(f"MinMaxScaler loaded from {scaler_filename}")
    scaled_data = MM.transform([data])
    Xdata = np.reshape(scaled_data,(1, 2381))
    Xdata= Xdata.tolist()
    return Xdata

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("binaries", metavar="BINARIES", type=str, nargs="+", help="PE files to classify")
  args = parser.parse_args()

  # print(args)

  # Load the pre-trained model
  model = MalConv()
  # Load the weights
  model.load_state_dict(torch.load("/gdrive/MyDrive/vMalConv/malConv_model_weights.pt", map_location=torch.device('cpu')))

  # Move model to GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  model.eval()

  # print("Loaded model from disk")

  data = feature_vector(open(args.binaries[0], 'rb').read())

  X_test_tensor = torch.tensor(data, dtype=torch.int)

  with torch.no_grad():
    inputs = X_test_tensor.to(device)  # Move inputs to the same device as model
    outputs = model(inputs)
    predicted = torch.round(outputs.squeeze())  # Convert probabilities to binary predictions (0 or 1)

    prediction = predicted.cpu().numpy()
    # print(prediction)
    if prediction > 0.5:
      print("It's a Malware")
      return "It's a Malware"
    else:
      print("It's a Benign Software")
      return "It's a Benign Software"

  

if __name__ == "__main__":
  main()

