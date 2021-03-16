import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class PricePredictModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        h = self.fc1(x)
        return self.fc2(h)


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        h = self.fc1(x)
        return self.fc2(h)


class TFJDRL(nn.Module):
    def __init__(self, temporal_dimension):
        """
        Args:
            temporal_dimension (int) : 
        """
        super().__init__()

        self.gated_fc = nn.Linear(5, 5)
        self.gru = nn.GRU(5, 5, num_layers=3)

        self.temporal_dimension = temporal_dimension
        self.temporal_h = []

        self.price_predictor = PricePredictModule(5*2, 5)
        self.policy_net = PolicyNet(5*2, 10)


    def __call__(self, x):
        self.temporal_h = []

        for i in range(self.temporal_dimension):
            f = x[i]
            g = self.gated_fc(f)
            h = F.glu(torch.cat([f, g]))
            if i != self.temporal_dimension - 1:
                self.temporal_h.append(h)

        mean = torch.zeros_like(x[0])# 仮　本来はここでアテンションを取る
        for th in self.temporal_h:
            mean += th
        mean /= len(x)

        env_vec = torch.cat([mean, h]) # 現在hには最後のステップの値が格納されている

        predicted_price = self.price_predictor(env_vec)
        action = self.policy_net(env_vec)

        print(predicted_price)
        print(action)

        return predicted_price, action


model = TFJDRL(2)

model(torch.tensor([[1.0,2.0,3.0,4.0,5.0],[2.0,2.0,3.0,4.0,5.0]]))
# model(torch.tensor())