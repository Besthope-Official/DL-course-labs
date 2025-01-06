import torch
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from time import time


class Trainer:

    def __init__(self, model, optimizer, criterion, train_dataset):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataset = train_dataset

        # use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        self.callbacks = defaultdict(list)
        # record each iteration time
        self.iter_time = 0
        self.iter_num = 0
        self.iter_dt = 0


    def set_callback(self, onevent: str, callback_func):
        self.callbacks[onevent] = [callback_func]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, criterion = self.model, self.criterion

        train_loader = DataLoader(
            self.train_dataset,
            # add a config to set DataLoader hyperparameters
            batch_size=128,
            num_workers=4,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time()

        for i, (x, y) in enumerate(train_loader):
            x, y = map(lambda t: t.to(self.device), (x, y))

            outputs = model(x)
            self.loss = criterion(outputs, y)

            # 梯度清零
            model.zero_grad(set_to_none=True)
            # 反向传播
            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            self.optimizer.step()

            # on batch end
            if (i + 1) % 100 == 0:
                self.trigger_callbacks('on_batch_end')
                self.iter_num = i
                tnow = time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow


def batch_end_callback(trainer):
    print(f'Iter {trainer.iter_num}, training time { \
          trainer.iter_dt:.2f}s: train loss {trainer.loss:.4f}')


## Usage of dependent functions ##
# if __name__ == '__main__':

#     trainer = Trainer()

#     trainer.set_callback('on_batch_end', batch_end_callback)
#     trainer.run()
