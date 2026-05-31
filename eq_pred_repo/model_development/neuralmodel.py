import sys
from pathlib import Path
import importlib.util
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import joblib
import helpers.feature_extract as fx
import helpers.load_data as ld
import helpers.evaluate as eh
import helpers.baseline_models as bsm
from add_func import decode_predictions

class NeuralModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc_T1 = nn.Linear(hidden_size, output_size)
        self.fc_T2 = nn.Linear(hidden_size, output_size)
        self.fc_T3 = nn.Linear(hidden_size, output_size)


    def attention(self, h):
        # h: [batch, seq_len, hidden]
        scores = self.attn(h).squeeze(-1)    
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * h).sum(dim=1) 
        return context

    def forward(self, X):
        X = X.contiguous()
        h, _ = self.lstm(X)                     
        context = self.attention(h)       

        T1 = self.fc_T1(context)
        T2 = self.fc_T2(context)
        T3 = self.fc_T3(context)

        out = torch.stack([T1, T2, T3], dim=1)
        delta_mag = out[:, :, 0]
        norm_time = torch.sigmoid(out[:, :, 1])
        return torch.stack([delta_mag, norm_time], dim=-1)
    
    def fit(self, X_list, Y_list,device,save_path,
            lr=1e-3, epochs=20):
        """"Assume X_list and Y_list is already scaled"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()

        optimizer = Adam(self.parameters(), lr=lr)

        criterion = nn.L1Loss()
        loss_list = []

        best_loss = float("inf")
        for i in tqdm(range(epochs), desc='Training', disable=False):
            loss_batch = []

            for df_X, df_Y in zip(X_list, Y_list):

                X = torch.tensor(df_X, dtype=torch.float32, device=device).unsqueeze(0)
                Y = torch.tensor(df_Y, dtype=torch.float32, device=device).unsqueeze(0)

                Y_pred = self.forward(X)
                loss = criterion(Y_pred, Y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                loss_batch.append(loss.item())

            epoch_loss = float(np.mean(loss_batch))
            loss_list.append(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if save_path is not None:
                    self.save(save_path)
            if i % 10 == 0:
                print(f'Epoch {i+1}/{epochs}, Loss: {loss_list[-1]:.4f}')
        return loss_list
    
    def predict_raw(self, df_X, device):
        """"
        Assume df_X is already scaled and changed to current features used
        Output will be change of Mag + normalized time diff
        """
        if device is None: device = next(self.parameters()).device
        self.eval()
        X = torch.tensor(df_X, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            pred = self.forward(X)
        return pred.squeeze(0).cpu().numpy()
    
    def predict(self, df_X, df_X_raw, device):
        return decode_predictions(self.predict_raw(df_X, device=device), df_X_raw)
    
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                    "input_size": self.lstm.input_size,
                    "hidden_size": self.lstm.hidden_size}, path)
        
    @classmethod
    def load(cls, path, device):
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=True)
        model = cls(input_size=ckpt["input_size"], hidden_size=ckpt["hidden_size"])
        model.load_state_dict(ckpt["state_dict"])
        return model