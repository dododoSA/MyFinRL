import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class PricePredictModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # 論文でいうところのaの値
        )

    def __call__(self, x):
        return self.seq(x)


class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh() # -1～1 に変換
        )

    def __call__(self, x):
        return self.seq(x)

class AttnModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, h_for_attn):
        attn_ene = self.seq(h_for_attn) # (temporal_dim-1, 1)
        return F.softmax(attn_ene, dim=0) 

class TFJDRL(nn.Module):
    def __init__(
        self,
        temporal_dimension,
        feature_size,
        gru_num_hidden,
        pp_num_hidden,
        policy_num_hidden,
        env_vec_size,
        attn_num_hidden=None
    ):
        """
        Args:
            temporal_dimension (int) : 系列長　何ステップ分過去のデータを考慮するか
            feature_size (int)       : 特徴量の次元
            gru_num_hidden (int)     : GRUの隠れ層のユニット数
            pp_num_hidden (int)      : 価格の予測モジュールの隠れ層のユニット数
            policy_num_hidden (int)  : 強化学習モジュールの隠れ層のユニット数
            attn_num_hidden (int)    : アテンション機構の隠れ層のユニット数
        """
        super().__init__()

        self.gated_fc = nn.Linear(feature_size, feature_size)
        self.gru = nn.GRU(feature_size, gru_num_hidden, num_layers=1, batch_first=True)

        self.temporal_dimension = temporal_dimension
        self.temporal_h = []

        self.price_predictor = PricePredictModule(env_vec_size, pp_num_hidden)
        self.policy_net = PolicyNet(env_vec_size, policy_num_hidden)

        self.fc = nn.Linear(gru_num_hidden*2, env_vec_size)

        self.attn_num_hidden = attn_num_hidden
        if not attn_num_hidden is None:
            self.attn = AttnModule(gru_num_hidden, attn_num_hidden)


    def __call__(self, x):
        f_prime = []

        # gate_structure
        for i in range(self.temporal_dimension):
            f = x[i]
            g = self.gated_fc(f)
            f_prime.append(F.glu(torch.cat([f, g])))
        f_prime = torch.stack(f_prime).unsqueeze(0)
        h, _ = self.gru(f_prime)
        h = h.squeeze() # 1ステップごとに実行するのでバッチサイズは1固定

        # T-1以前に対してアテンションをとる

        if self.attn_num_hidden is None:
            h_T = h[-1].repeat(len(h[:-1]), 1)
            tmp = (h_T*h[:-1]).sum(dim=1)
            attns = F.softmax(torch.tanh(tmp), dim=0)
            attns = attns.unsqueeze(1)
            attn_vec = (h[:-1] * attns).sum(dim=0)
        else:
            attns = self.attn(h[:-1])
            attn_vec = (h[:-1] * attns).sum(dim=0)

        # T とアテンションをとったT-1を合わせてenv_vecとする

        env_vec = torch.tanh(self.fc(torch.cat([attn_vec, h[-1]])))

        # 株価の予測モジュールと強化学習モジュール
        predicted_price = self.price_predictor(env_vec)
        action = self.policy_net(env_vec)

        print(predicted_price)
        print(action)

        return predicted_price, action

