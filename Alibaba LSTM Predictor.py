from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import time
from AlibabaLSTMModule import AlibabaLSTM
from Trainer import Trainer
from CustomDataClass import dataset_for_time_series, DataModule
from sklearn.preprocessing import StandardScaler
import argparse
import json

######################################
# Set an option whether Hyperparameter Optimization should be included
######################################

parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument("--hyper", action="store_true", help="Use Hyperparameter optimization")

args = parser.parse_args()

hyperparameter_optimization = args.hyperparameter_optimization

if hyperparameter_optimization:
    print("Hyperparameter optimization:", hyperparameter_optimization)

##################################
# Load the dataframe
##################################

print('Load the dataframe')
workload_dataframe = pd.read_csv('alibaba_workload_data.csv')
workload_dataframe = workload_dataframe.fillna(0)


standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(workload_dataframe)
workload_data = pd.DataFrame(standardized_data, columns=workload_dataframe.columns)

# Scale data
workload_data *= 10

# Select relevant columns
workload_data = workload_data[['avg_cpu', 'avg_gpu', 'avg_memory', 'avg_new_cpu', 'avg_new_gpu', 'avg_new_memory']]
input_size = workload_data.shape[1]

############################################
# Do the optimization loop:
############################################

if hyperparameter_optimization:
    print('Hyperparameter Optimization Loop')
    best_params, best_accuracy = Trainer.hyperparameter_optimization(workload_data, input_size, 3, 3)
    lstm_model = AlibabaLSTM(input_size, best_params['hidden_size'], 3, best_params['hidden_size_2'], best_params['hidden_size_3'], 3,
                          num_layers=best_params['num_layers'], optimizer=best_params['optimizer'], learning_rate=best_params['learning_rate'],
                          loss_function=best_params['loss'], clip_val=best_params['gradient_clip'], scheduler=best_params['scheduler'])
    trainer = Trainer(50, best_params['batch_size'], early_stopping_patience=6, window_size=best_params['window_size'])
else:
    lstm_model = AlibabaLSTM(input_size, 128, 3, 32, 8, 3, num_layers=2, optimizer='Adam', learning_rate=0.0001,
                          loss_function='Huber', scheduler='OnPlateau', l1=0.0, clip_val=0.0, l2=0.0)
    trainer = Trainer(40, 128, early_stopping_patience=6, window_size=18)

trainer.fit(lstm_model, workload_data)

######################################
# Visualize Results
######################################

n_epochs = range(trainer.max_epochs)
train_loss = trainer.train_loss_values
nan_values = np.full(trainer.max_epochs - len(train_loss), np.nan)
train_loss = np.concatenate([train_loss, nan_values])

val_loss = trainer.val_loss_values
nan_values = np.full(trainer.max_epochs - len(val_loss), np.nan)
val_loss = np.concatenate([val_loss, nan_values])

plt.figure(figsize=(10, 6))
plt.plot(n_epochs, train_loss, color='blue', label='Train Loss', linestyle='-')
plt.plot(n_epochs, val_loss, color='orange', label='Validation Loss', linestyle='-')
plt.legend()
plt.show()

######################################
# Test the model
######################################

y_hat, y = trainer.test(lstm_model)

y_0 = y[:, 0]
y_1 = y[:, 1]
y_2 = y[:, 2]
y_hat_0 = y_hat[:, 0]
y_hat_1 = y_hat[:, 1]
y_hat_2 = y_hat[:, 2]

x_values = range(len(y_0))

plt.figure()
plt.plot(x_values, y_0, color='blue', label='CPU Usage', linestyle='-')
plt.plot(x_values, y_hat_0, color='green', label='Prediction', linestyle='-')
plt.legend()
plt.show()

plt.figure()
plt.plot(x_values, y_1, color='red', label='GPU Usage', linestyle='-')
plt.plot(x_values, y_hat_1, color='green', label='Prediction', linestyle='-')
plt.legend()
plt.show()

plt.figure()
plt.plot(x_values, y_2, color='orange', label='Memory Usage', linestyle='-')
plt.plot(x_values, y_hat_2, color='green', label='Prediction', linestyle='-')
plt.legend()
plt.show()



