import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from models import GCNLinkPred, SAGELinkPred
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


def hit_k(ranks, k):
    hits = sum(1.0 for rank in ranks if rank <= k)
    return hits / len(ranks)


def mrr(ranks):
    reciprocal_ranks = [1.0 / rank for rank in ranks]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        scores = model(data.x, data.edge_index,
                       data.edge_label_index).cpu().numpy()
        labels = data.edge_label.cpu().numpy()
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        ranks = []
        for idx in pos_indices:
            score = scores[idx]
            higher_neg = (scores[neg_indices] > score).sum()
            rank = higher_neg + 1
            ranks.append(rank)
        hit100 = hit_k(ranks, 100)
        mrr_value = mrr(ranks)
        return hit100, mrr_value


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]

    split = RandomLinkSplit(num_val=0.1, num_test=0.2,
                            is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = split(data)

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    model = GCNLinkPred(
        dataset.num_features,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    num_epochs = 200

    best_val_hit100 = 0
    best_model_state = None

    train_losses = []
    val_losses = []
    epochs = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(train_data.x, train_data.edge_index,
                       train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(
            scores, train_data.edge_label)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_losses.append(train_loss)
        epochs.append(epoch + 1)

        model.eval()
        with torch.no_grad():
            val_scores = model(val_data.x, val_data.edge_index,
                               val_data.edge_label_index)
            val_loss = F.binary_cross_entropy_with_logits(
                val_scores, val_data.edge_label).item()
            val_losses.append(val_loss)
            val_hit100, val_mrr = evaluate(model, val_data)

            if val_hit100 > best_val_hit100:
                best_val_hit100 = val_hit100
                best_model_state = model.state_dict().copy()

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Hit@100: {val_hit100:.4f}, Val MRR: {val_mrr:.4f}')
        scheduler.step(val_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    print('Loss curve saved to loss_curve.png')

    model.load_state_dict(best_model_state)

    test_hit100, test_mrr = evaluate(model, test_data)
    print(f'Test Hit@100: {test_hit100:.4f}, Test MRR: {test_mrr:.4f}')


if __name__ == '__main__':
    main()
