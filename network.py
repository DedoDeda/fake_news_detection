import torch
from torch import nn, optim
from torch.utils import data

from log import log

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Network(nn.Module):
    def __init__(self, num_embeddings, out_size, embedding_size):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.linear = nn.Linear(embedding_size, out_size)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_flat = embedded.mean(1)
        y_hat = self.linear(embedded_flat)
        return y_hat


def create_network(num_embeddings, out_size, embedding_size):
    return Network(num_embeddings, out_size, embedding_size).to(DEVICE)


def load_network(num_embeddings, out_size, embedding_size, state_path):
    network = create_network(num_embeddings, out_size, embedding_size)
    network.load_state_dict(torch.load(state_path))
    return network


def train(network, dataset, batch_size, shuffle_dataset, learning_rate, num_epochs):
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_dataset,
        collate_fn=dataset.collate_fn)
    optimizer = optim.Adam(network.parameters(), learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    log('Starting to train network...')

    for epoch in range(num_epochs):
        total_losses = 0
        correct_evals = 0
        total_evals = 0

        for batch in data_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_hat = network(x)
            loss = loss_fn(y_hat, y)

            total_losses += loss * y.size(0)
            total_evals += y.size(0)
            correct_evals += (y_hat.argmax(1) == y.argmax(1)).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log(f'Epoch {epoch + 1}/{num_epochs}: mean loss = {total_losses / total_evals:.3}, '
            f'accuracy = {correct_evals / total_evals:.3%}.')

    log('Finished training network.')


def save_network(network, state_path):
    with open(state_path, 'wb') as file:
        torch.save(network.state_dict(), file)


def test(network, dataset, batch_size, shuffle_dataset):
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_dataset,
        collate_fn=dataset.collate_fn)
    loss_fn = nn.CrossEntropyLoss()

    log('Starting to test network...')

    total_losses = 0
    correct_evals = 0
    total_evals = 0

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_hat = network(x)
            loss = loss_fn(y_hat, y).item()

            total_losses += loss * y.size(0)
            total_evals += y.size(0)
            correct_evals += (y_hat.argmax(1) == y.argmax(1)).sum().item()

    log('Finished testing network.')
    log(f'Test: mean loss = {total_losses / total_evals:.3}, accuracy = {correct_evals / total_evals:.3%}.')
