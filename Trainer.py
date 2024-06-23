import torch
import torch.nn as nn
import time
from CustomDataClass import dataset_for_time_series, DataModule
import optuna
from AlibabaLSTMModule import AlibabaLSTM


class Trainer:
    def __init__(self, max_epochs, batch_size=8, early_stopping_patience=6, min_delta=0.09, num_gpus=0, window_size=10):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.best_val_loss = float('inf')
        self.num_epochs_no_improve = 0
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_training_data(self, workload_data, batch_size):
        split_index = int(len(workload_data) * 0.8)  # 80% train, 10% val, 10% test
        split_index2 = int(split_index + ((len(workload_data) - split_index) / 2))

        print('Prepare the labels')
        train_data_X, train_data_y = dataset_for_time_series(workload_data[:split_index], self.window_size)
        val_data_X, val_data_y = dataset_for_time_series(workload_data[split_index:split_index2], self.window_size)
        test_data_X, test_data_y = dataset_for_time_series(workload_data[split_index2:-2], self.window_size)

        print('Prepare the train_dataset')
        data_train = DataModule(train_data_X, train_data_y)

        print('Prepare the val_dataset')
        data_val = DataModule(val_data_X, val_data_y)

        print('Prepare the test_dataset')
        data_test = DataModule(test_data_X, test_data_y)

        self.train_dataloader = data_train.get_dataloader(batch_size)
        self.val_dataloader = data_val.get_dataloader(batch_size)
        self.test_dataloader = data_test.get_dataloader(batch_size)

    def prepare_test_data(self, data_test):
        test_data_X, test_data_y = dataset_for_time_series(data_test, self.window_size)
        data_test = DataModule(test_data_X, test_data_y)
        self.test_dataloader = data_test.get_dataloader(self.batch_size)

    def prepare_model(self, model):
        model.trainer = self
        self.model = model

    def fit(self, model, training_and_validation_data):
        self.train_loss_values = []
        self.val_loss_values = []
        self.prepare_training_data(training_and_validation_data, self.batch_size)
        self.prepare_model(model)

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss, val_loss = self.fit_epoch()

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch + 1}/{self.max_epochs}], Train_Loss: {train_loss:.4f}, Val_Loss: {val_loss:.4f}, '
                      f'LR = {self.model.scheduler.get_last_lr() if self.model.scheduler else self.model.learning_rate}')

            self.train_loss_values.append(train_loss)
            self.val_loss_values.append(val_loss)

            # Early Stopping
            if (self.best_val_loss - val_loss) > self.min_delta:
                self.best_val_loss = val_loss
                self.num_epochs_no_improve = 0
            else:
                self.num_epochs_no_improve += 1
                if self.num_epochs_no_improve == self.early_stopping_patience:
                    print("Early stopping at epoch", epoch)
                    break

            # Scheduler step
            if self.model.scheduler:
                self.model.scheduler.step(val_loss)

    def fit_epoch(self):
        train_loss = 0.0
        total_batches = len(self.train_dataloader)

        for idx, (x_batch, y_batch) in enumerate(self.train_dataloader):
            additional = y_batch[:, -1, 3:]
            output = self.model(x_batch, additional)
            loss = self.model.loss(output, y_batch[:, -1, 0:3])
            self.model.optimizer.zero_grad()
            loss.backward()

            # L1 Loss
            if self.model.l1_rate != 0:
                loss += self.model.l1_regularization(self.model.l2_rate)

            # Gradient Clipping
            if self.model.clip_val != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.clip_val)

            self.model.optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

            # Simulate batch processing time
            time.sleep(0.1)

            # Calculate progress
            progress = (idx + 1) / total_batches * 100
            print(f"\rBatch {idx + 1}/{total_batches} completed. Progress: {progress:.2f}%", end='', flush=True)

        train_loss /= len(self.train_dataloader.dataset)
        val_loss = self.evaluate(self.val_dataloader)
        return train_loss, val_loss

    def evaluate(self, dataloader):
        val_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                additional = y_batch[:, -1, 3:]
                val_output = self.model(x_batch, additional)
                loss = self.model.loss(val_output, y_batch[:, -1, 0:3])
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= len(dataloader.dataset)
        return val_loss

    def test(self, model, data_test=None):
        model.eval()
        self.test_loss = 0.0

        if data_test is not None:
            self.prepare_test_data(data_test)

        y_hat_total = torch.zeros(1, 3)
        y_total = torch.zeros(1, 3)

        with torch.no_grad():
            for X, y in self.test_dataloader:
                additional = y[:, -1, 3:]
                y_hat = model(X, additional)
                y_total = torch.cat((y_total, y[:, -1, 0:3]), dim=0)
                y_hat_total = torch.cat((y_hat_total, y_hat), dim=0)
                loss = self.model.metric(y_hat, y[:, -1, 0:3])
                self.test_loss += loss * X.size(0)

        self.test_loss /= len(self.test_dataloader.dataset)
        return y_hat_total[1:], y_total[1:]

    @classmethod
    def optuna_objective(cls, trial, workload_data, input_size, concat_size, output_size):
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        hidden_size_2 = trial.suggest_categorical('hidden_size_2', [16, 32])
        hidden_size_3 = trial.suggest_categorical('hidden_size_3', [8, 32])
        l2_rate = trial.suggest_categorical('l2_rate', [0.0, 0.0001, 0.005])
        loss_function = trial.suggest_categorical('loss', ['MSE', 'Huber'])
        window_size = trial.suggest_categorical('window_size', [10, 15, 20])
        gradient_clip = trial.suggest_categorical('gradient_clip', [0.0, 1.0])
        scheduler = trial.suggest_categorical('scheduler', [None, 'OnPlateau'])
        num_layers = trial.suggest_categorical('num_layers', [1, 2])

        model = AlibabaLSTM(
            input_size, hidden_size, concat_size, hidden_size_2, hidden_size_3, output_size,
            num_layers, learning_rate=learning_rate, optimizer=optimizer, loss_function=loss_function,
            clip_val=gradient_clip, scheduler=scheduler
        )
        trainer = cls(40, batch_size, window_size=window_size)
        trainer.fit(model, workload_data)
        return trainer.val_loss_values[-1]

    @classmethod
    def hyperparameter_optimization(cls, workload_data, input_size, concat_size, output_size):
        study = optuna.create_study(direction='minimize')
        objective_func = lambda trial: cls.optuna_objective(trial, workload_data, input_size, concat_size, output_size)
        study.optimize(objective_func, n_trials=30)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_accuracy = best_trial.value
        return best_params, best_accuracy
