from tqdm import tqdm
import torch

class Trainer:
    def __init__(self, loss_fn,  device, batch_size=64, input_dim=784, num_class=10):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_class = num_class
        self.device = device
        self.loss_fn = loss_fn

    def train(self, train_data, net, optim, num_epochs = 10):
        loss_val = []
        acc_val = []
        total_data = len(train_data)
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            acc = 0
            for x, y in train_data:
                x = x.to(torch.float32).to(self.device)
                
                if (len(x.shape)==4):
                    x = x.mean(dim=1)
                x = x.view(-1, self.input_dim)
                y_pred = net(x)

                y_class = torch.zeros(y.shape[0], self.num_class)
                y_class[torch.arange(y.shape[0]), y] = 1
                y_class = y_class.to(torch.float32).to(self.device)

                loss = self.loss_fn(y_class, y_pred)

                total_loss += loss.item()
                with torch.no_grad():
                    acc += ((y_pred.argmax(dim =-1) - y.to(self.device)) == 0).to(torch.float).sum()

                optim.zero_grad()
                loss.backward()
                optim.step()
            acc = acc/(self.batch_size * total_data)
            total_loss = total_loss/total_data
            print(f'Epoch {epoch} with Loss {total_loss} and accuracy {acc}')
            loss_val.append(total_loss)
            acc_val.append(acc)
        return loss_val, acc_val

    def eval(self, model, test_data):
        acc = 0
        for x, y in test_data:
            x = x.to(torch.float32).to(self.device)
            y = y.to(torch.float32).to(self.device)
            x = x.view(-1, self.input_dim)
            with torch.no_grad():
                y_pred = model(x)
            acc += ((y_pred.argmax(dim =-1) - y) == 0).to(torch.float).sum()
        return 100*acc/(64*len(test_data))
