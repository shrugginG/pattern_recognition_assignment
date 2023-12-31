from utils.compute_device import COMPUTE_DEVICE
from models.gcn import GCN_2
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.loader import DataLoader
import torch
from torch.optim import Adam

from dataset.dataset_v2 import PhishingDataset2

preprocessed_file_path = './data/train'
phishing_dataset = PhishingDataset2(
    root=preprocessed_file_path, do_data_preparation=False)


model = GCN_2(in_channels=phishing_dataset.num_features,
              hidden_channels=16,
              out_channels=phishing_dataset.num_classes,
              pooling_fn=global_mean_pool,
              device=COMPUTE_DEVICE).to(COMPUTE_DEVICE)

train_dataset = phishing_dataset[:2781]
test_dataset = phishing_dataset[2781:3707]
val_dataset = phishing_dataset[3707:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def eval_loss(model: torch.nn.Module, los_fun):
    model.eval()
    loss = 0
    for data in val_loader:
        data.to(COMPUTE_DEVICE)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
        loss += los_fun(out, data.y) * data.num_graphs
    return loss/len(val_loader.dataset)


def eval_acc(model: torch.nn.Module):
    model.eval()

    correct = 0
    for data in test_loader:
        data.to(COMPUTE_DEVICE)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.batch).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct/len(test_loader.dataset)


def train(model, loss_fun, epochs, lr, weight_decay):

    total_acc = 0

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(COMPUTE_DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            loss = loss_fun(out, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # fixme
            total_loss += float(loss) * data.num_graphs
        loss = total_loss / len(train_loader.dataset)
        test_acc = eval_acc(model)
        total_acc += test_acc
        print(
            f"epoch: {epoch}, train_loss:{loss}, val_loss:{eval_loss(model,loss_fun)}, test_acc:{test_acc}")
    print(f"total_acc: {total_acc/epochs}")


train(model, torch.nn.functional.cross_entropy, 50, 0.01, 4e-5)
