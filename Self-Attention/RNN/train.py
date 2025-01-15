import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import NameDataset, phrase2list, make_tensors
from RNN import RNNClassifier
from config import device, HIDDEN_SIZE, NUM_CHARS, NUM_LAYERS, drop_out, BATCH_SIZE, NUM_EPOCHS, patience


train_set = NameDataset(r'data/train.tsv', train=True)
val_set = NameDataset(r'data/train.tsv', val=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
NUM_CLASS = len(set(train_set.sentiment))


def train_and_validate(model, criterion, optimizer, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for phrase, sentiment in train_loader:
        inputs, seq_lengths, target = make_tensors(phrase, sentiment)
        inputs, seq_lengths, target = inputs.to(
            device), seq_lengths.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(inputs, seq_lengths)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for phrase, sentiment in val_loader:
            inputs, seq_lengths, target = make_tensors(phrase, sentiment)
            inputs, seq_lengths, target = inputs.to(
                device), seq_lengths.to(device), target.to(device)
            output = model(inputs, seq_lengths)
            loss = criterion(output, target)
            val_loss += loss.item()
    return total_loss / len(train_loader), val_loss / len(val_loader)


def return_classifier():
    classifier = torch.load('sentimentAnalyst.pkl')
    classifier.to(device)
    return classifier


if __name__ == '__main__':
    classifier = RNNClassifier(
        NUM_CHARS, HIDDEN_SIZE, NUM_CLASS, drop_out, NUM_LAYERS).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    loss_list = []
    patience_counter = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, val_loss = train_and_validate(
            classifier, criterion, optimizer, train_loader, val_loader, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        loss_list.append(val_loss)
        if val_loss <= min(loss_list):
            torch.save(classifier, 'sentimentAnalyst.pkl')
            print('Save Model!')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('早退')
                break
    return_classifier()
    epoch = [epoch + 1 for epoch in range(len(loss_list))]
    plt.plot(epoch, loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
