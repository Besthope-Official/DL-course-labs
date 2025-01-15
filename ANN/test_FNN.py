from FNN import MLP
from CNN import CNN
from utils import Trainer, batch_end_callback
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score


torch.manual_seed(42)

def load_data():
    # prepare the data
    # convert image to tensor and normalized
    train_data = datasets.MNIST(
        root='data', train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(
        root='data', train=False, transform=transforms.ToTensor(), download=True)
    print('train_data size:', train_data.data.size())
    print('test_data size:', test_data.data.size())
    return train_data, test_data


def test(model, test_data, criterion):
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    test_loss = []
    all_preds = []
    all_targets = []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
            test_loss.append(loss.item())
            preds = pred.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            targets = y.cpu().numpy()
            all_targets.extend(targets.tolist())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(y)
    test_loss = sum(test_loss) / len(test_loss)
    accuracy = correct / total

    prec = precision_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    return test_loss, accuracy, prec, f1


if __name__ == '__main__':
    train_data, test_data = load_data()
    # CNN on MNIST (baseline)
    # model = CNN()
    # MLP on MNIST
    model = MLP(activation='sigmoid')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, optimizer, criterion, train_data)
    trainer.set_callback('on_batch_end', batch_end_callback)

    epoch_num = 8
    for epoch in range(epoch_num):
        trainer.run()
        test_loss, accuracy, prec, f1 = test(model, test_data, criterion)
        print(f'Epoch [{epoch+1}]/[{epoch_num}] Test Loss: {
              test_loss:.4f}, Test Accuracy: {accuracy:.4f}, prec: {prec:.4f}, f1: {f1:.4f}')

    # Visualize some examples
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    plt.figure(figsize=(6, 3), constrained_layout=True)
    for i in range(12):
        plt.subplot(2, 6, i+1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title("P: %d, L: %d" % (predicted[i], labels[i]))
    plt.show()
