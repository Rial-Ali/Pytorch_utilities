# Import packages
import torch
import time
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("cuda")
else:
    device = torch.device('cpu')
    print("cpu")


class MyModel():
    '''A Class use as wraper to pytorch model.

    The Class train the model with monitoring the fit state. It use validation data to stop training when the model
    start overfiting. While training the Class save the best model weights.

    Attributes
    ----------
    model : pytorch model
        model that you want to train.
    loss_func : loss function object
        loss function use to train model
    optimizer : optimizer object
        optimizer use to train model

    Methods
    -------
    train(epochs, train_data, val_data, val_wait=10)
        train the model on given train data and validation on given validation data.
    '''

    def __init__(self, model, loss_func, optimizer):
        '''

        Parameters
        ----------
        model : pytorch model
            model that you want to train.
        loss_func : loss function object
            loss function use to train model
        optimizer : optimizer object
            optimizer use to train model
        '''

        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.old_val_loss = 10000 # Save the latest validation loss.
        self.best_model_val = 10000 # Save the best validation loss the model obtained.
        self.best_model = None # Save the model that obtained the best validation loss.
        self.epoch = 0  # The Number of epochs the training start from.

    def train(self, train_data, val_data, epochs=100, val_wait=10):
        '''Train the model on given train data and validation on given validation data..

        After Training plot learninig curve.

        Parameters
        ----------
        train_data : torch tensor
            Data to train the model on
        val_data : torch tensor
            Data to validate the model on
        epochs : int, optioal
            The number of epochs to train the model (default is 100)
        val_wait : int, optional
            The number of epochs to train before stop the training when the validation loss begin to increase
            (default is 10)
        '''

        stamp = time.time()

        train_losses = []
        val_losses = []
        train_loss = 0.0
        val_step = 0

        # Training loop
        for epoch in tqdm(range(self.epoch, self.epoch + epochs)):
            for i, sample in enumerate(train_data):
                x = sample[0].to(device)
                y = sample[1].to(device)

                output = self.model(x)

                self.optimizer.zero_grad()
                loss = self.loss_func(output, y)
                train_loss += loss
                loss.backward()
                self.optimizer.step()

            val_loss = self.test(val_data)
            # If the model does not improve on validation data start the counter of epoch to stop after val_wait epoch.
            if self.old_val_loss < val_loss:
                val_step += 1
            else:
                val_step = 0


            train_loss = train_loss.to('cpu') / len(train_data)
            train_losses.append(np.squeeze(train_loss.detach().numpy()))

            val_loss = val_loss.to('cpu') / len(val_data)
            val_losses.append(val_loss.detach().numpy())

            self.old_val_loss = val_loss
            train_loss = 0.0
            print(f'epoch {epoch+1}: train_loss = {train_loss:.8f}, val_loss = {val_loss:.8f}')

            # If actual model is better the previous we saved, save this model instead of it.
            if val_loss < self.best_model_val:
                self.best_model_val = val_loss
                self.best_model = copy.deepcopy(self.model)
                torch.save({"model_state_dict": self.best_model.state_dict(),
                            "epoch": epoch + 1,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss_func": self.loss_func,
                            "best_model_val": self.best_model_val},
                           f"Model--epoch-{epoch+1}--val_loss-{self.best_model_val}.pth")
                print(f"saving model.... model_val loss {self.best_model_val}")

            # If the model since a specific epochs count did not improve then stop the training process.
            if val_step >= val_wait:
                print(
                    f"Break because the val_loss not decreasing since {val_step} epoch, "
                    f"the best model achieve {self.best_model_val} on validation data.")
                break



        self.epoch = epoch + 1

        plt.plot(range(len(train_losses)), train_losses, label='train_loss')
        plt.plot(range(len(val_losses)), val_losses, label='val_loss')
        plt.title("Training_curve")
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.savefig(f"train{epoch}--{str(stamp)}.png")
        plt.show()
        print('Training is done !!')

    def test(self, test_data):
        '''Test the model on the given data

        Parameters
        ----------
        test_data : torch tensor
            Data to test the model on

        returns
        -------
        test_loss : float
            Loss is obtained on the given data
        '''

        test_loss = 0.0

        with torch.no_grad():
          for i, sample in enumerate(test_data):
            x = sample[0].to(device)
            y = sample[1].to(device)

            output = self.model(x)
            loss = self.loss_func(output, y)
            test_loss += loss

        return  test_loss

    def load_model(self, model_name):
        '''

        Parameters
        ----------
        model_name : str
             The name of file to load the model from.
        '''

        checkpoint = torch.load(model_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss_func = checkpoint['loss_func']
        self.best_model_val = checkpoint['best_model_val']