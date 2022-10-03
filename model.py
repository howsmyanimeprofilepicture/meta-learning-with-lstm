import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score


class MANN(nn.Module):
    def __init__(self, num_classes, num_samples, embed_size=784):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.embed_size = embed_size
        self.lstm1 = nn.LSTM(
            embed_size + num_classes, 128, batch_first=True
        )  # Pytorch always has return_sequences=True
        self.lstm2 = nn.LSTM(128, num_classes, batch_first=True)

    def forward(self, inputs):
        x, _ = self.lstm1(inputs)
        x, _ = self.lstm2(x)

        return x[:, -self.num_classes :, :]


def train_model(
    model: nn.Module,
    meta_inputs,  # (-1, batch_size, (K-1)*N, d)
    meta_labels,  # (-1, batch_size, N)
    criterion,
    optim,
    epoch: int = 2500,
    disp: int = 50,
    cuda=False
):  
    NUM_OF_BATCHES = len(meta_inputs)
    last_loss = 0.
    model.train()
    for i in range(epoch):
        for j, (input, label) in enumerate(zip(meta_inputs, meta_labels)):
            optim.zero_grad()
            input = torch.tensor(input)
            y_true = torch.tensor(label)
            if cuda:
                input = input.to(device="cuda")
                y_true = y_true.to(device="cuda")
            y_pred = model(input)
            loss = criterion(y_pred, y_true)
            if j == NUM_OF_BATCHES-1:
                last_loss = loss.item()
            loss.backward()
            optim.step()

        if i % disp == disp - 1:
            print(f"Loss: {last_loss:.2f}")
    # return last_loss


@torch.no_grad()
def eval_model(
    model: nn.Module,
    meta_inputs,  # (-1, batch_size, (K-1)*N, d)
    meta_labels,  # (-1, batch_size, N)
    cuda=False,
):
    model.eval()
    preds = []
    for j, (input, label) in enumerate(zip(meta_inputs, meta_labels)):
        input = torch.tensor(input)
        y_true = torch.tensor(label)
        if cuda:
            input = input.to(device="cuda")
            y_true = y_true.to(device="cuda")
        logit = model(input)
        pred = torch.argmax(logit, axis=-1).cpu().reshape(-1).tolist()
        preds += pred

    print(f"Acc: {accuracy_score(meta_labels.reshape(-1), preds):.4f}", )
