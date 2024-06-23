from torch import nn, optim
import torch
import torch.nn.init as init
import torch.nn.functional as F

class AlibabaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, concat_size, hidden_size_2, hidden_size_3, output_size, 
                 num_layers=1, optimizer='SGD', learning_rate=0.001, loss_function='MSE', l1=0.0, l2=0.0, 
                 clip_val=0, scheduler=None):
        super(AlibabaLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.l1_rate = l1
        self.l2_rate = l2
        self.clip_val = clip_val
        self.learning_rate = learning_rate

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.initialize_weights(self.lstm, 'Xavier', 1)
        
        # Additional vector to concatenate
        self.fc_concat = nn.Linear(concat_size, hidden_size_2)
        self.initialize_weights(self.fc_concat, 'He', 0)
        
        # Fully connected layers
        self.fc = nn.Linear(hidden_size + hidden_size_2, hidden_size_3)
        self.initialize_weights(self.fc, 'Normal', 0)
        self.fc2 = nn.Linear(hidden_size_3, output_size)
        self.initialize_weights(self.fc2, 'Normal', 0)
        
        self.optimizer = self.get_optimizer(optimizer)
        self.loss = self.get_loss(loss_function)
        self.metric = self.get_metric()
        self.scheduler = self.get_scheduler(scheduler)
        
        self.activation = nn.PReLU()
        self.activation2 = nn.LeakyReLU()

    def forward(self, x, additional):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        final_hidden_state = h_n[-1]

        additional_transformed = self.activation(self.fc_concat(additional))
        concatenated = torch.cat((final_hidden_state, additional_transformed), dim=1)
        hidden = self.fc(concatenated)
        output = self.activation2(self.fc2(hidden))

        return output
    
    def get_optimizer(self, optimizer):
        optimizers = {
            'Adam': optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_rate),
            'SGD': optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.09, weight_decay=self.l2_rate)
        }
        return optimizers[optimizer]

    def l1_regularization(self, loss):
        l1_reg = sum(p.abs().sum() * self.l1_rate for p in self.parameters())
        loss += l1_reg
        return loss

    def get_loss(self, loss_function):
        loss_functions = {
            'CEL': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss(),
            'MAE': lambda y, y_hat: torch.mean(torch.abs(y - y_hat)),
            'Huber': nn.HuberLoss()
        }
        return loss_functions[loss_function]

    def get_metric(self):
        return lambda y, y_hat: torch.mean(torch.abs(y - y_hat))

    def get_scheduler(self, scheduler):
        if scheduler is None:
            return None
        schedulers = {
            'OnPlateau': optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.15, patience=5, threshold=0.0001)
        }
        return schedulers[scheduler]

    def initialize_weights(self, layer, initialization='Normal', bias=0):
        init_methods = {
            'Xavier': init.xavier_uniform_,
            'Uniform': lambda x: init.uniform_(x, a=-0.1, b=0.1),
            'Normal': lambda x: init.normal_(x, mean=0, std=0.01),
            'He': lambda x: init.kaiming_normal_(x, mode='fan_in', nonlinearity='relu')
        }

        init_method = init_methods[initialization]

        for name, param in layer.named_parameters():
            if 'weight' in name:
                init_method(param)
            elif 'bias' in name:
                nn.init.constant_(param, bias)

    def clip_gradients(self, clip_value):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-clip_value, clip_value)

