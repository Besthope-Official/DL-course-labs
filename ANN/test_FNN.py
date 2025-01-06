from FNN import MLP
from utils import Trainer, batch_end_callback
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def load_data():
    # prepare the data
    # convert image to tensor and normalized
    train_data = datasets.MNIST(
        root='data', train=True, transform=transforms.ToTensor(), download=False)
    test_data = datasets.MNIST(
        root='data', train=False, transform=transforms.ToTensor(), download=False)
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
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
            test_loss.append(loss.item())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(y)
    test_loss = sum(test_loss) / len(test_loss)
    accuracy = 100 * correct / total
    return test_loss, accuracy


if __name__ == '__main__':
    train_data, test_data = load_data()
    # MLP on MNIST
    model = MLP()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, optimizer, criterion, train_data)
    trainer.set_callback('on_batch_end', batch_end_callback)

    epoch_num = 10
    for epoch in range(epoch_num):
        trainer.run()
        test_loss, accuracy = test(model, test_data, criterion)
        print(f'Epoch [{epoch+1}]/[{epoch_num}] Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}%')
        
