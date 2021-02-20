import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df, 
                stock_dim,
                hmax,                
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                make_plots = False, 
                print_verbosity = 10,
                day = 0, iteration=''):
        # 学習データの何日目以降を使用するか（これを使用するならもともとのデータを加工した方がいいと思う）
        self.day = day
        # 学習データのDataFrame YahooDownloaderでダウンロードしたデータをFeatureEngineerで加工したものを渡す
        self.df = df
        # 銘柄の数
        self.stock_dim = stock_dim
        # 一度に取引が可能な最大数
        self.hmax = hmax
        # 投資にいくら使用するか
        self.initial_amount = initial_amount
        # 手数料(%)
        self.transaction_cost_pct =transaction_cost_pct
        # 報酬のスケール　学習の安定化のため？
        self.reward_scaling = reward_scaling
        # balance(1) + (銘柄ごとの株価) + (銘柄ごとの持ち株数) + (銘柄ごとのテクニカル指標)
        self.state_space = state_space
        # 銘柄数
        self.action_space = action_space
        # 使用するテクニカル指標
        self.tech_indicator_list = tech_indicator_list
        # 銘柄数の次元だけ[-1, 1]の行動空間を定義
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,)) 
        # 状態空間を定義
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        # dayで指定したその日の終値やテクニカル指標などのデータ
        self.data = self.df.loc[self.day,:]
        # 終了判定フラグ
        self.terminal = False
        # 資産の推移を画像として保存するかどうか
        self.make_plots = make_plots
        # どれくらいの間隔でログを出力するか　TODO:間隔の定義をしらべる
        self.print_verbosity = print_verbosity
        # turbulence_indexの閾値
        self.turbulence_threshold = turbulence_threshold
        # stateとして使用する[balance, 株価, 持ち株数, 各テクニカル指標]を生成
        self.state = self._initiate_state()
        
        # initialize reward
        # 報酬
        self.reward = 0
        # turbulence_index
        self.turbulence = 0
        # 手数料の累計
        self.cost = 0
        # 取引回数
        self.trades = 0
        # エピソード数
        self.episode = 0
        # memorize all the total balance change
        # 資産の記録
        self.asset_memory = [self.initial_amount]
        # 報酬の記録
        self.rewards_memory = []
        # 行動の記録
        self.actions_memory=[]
        # 日付の記録 self._get_date():現在のself.dataの日付
        self.date_memory=[self._get_date()]
        #self.reset()
        # ランダムシードを決定デフォルトではNone
        self._seed()


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence>self.turbulence_threshold:
                # if turbulence goes over threshold, just clear out all positions 
                # もし株を保有していたら
                if self.state[index+self.stock_dim+1] > 0:
                    #update balance
                    # そのタイミングでの株価で売却、手数料込み
                    self.state[0] += self.state[index+1]*self.state[index+self.stock_dim+1]* \
                                (1- self.transaction_cost_pct)
                    # 保有株式を0に
                    self.state[index+self.stock_dim+1] =0
                    # これみすってね？ self.state[index+self.stock_dim+1]が0だからどうせ0になる気がする ↑と↓ の行が逆?
                    self.cost += self.state[index+1]*self.state[index+self.stock_dim+1]* \
                                self.transaction_cost_pct
                    self.trades+=1
                else:
                    pass
        else:
            # perform sell action based on the sign of the action
            # 株式を保有していたら
            if self.state[index+self.stock_dim+1] > 0:
                #update balance
                # actionの株式数だけそのタイミングでの株価で売却、手数料込み actionって実数なんだね
                self.state[0] += \
                self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
                (1- self.transaction_cost_pct)
                # 保有株式数を減らす
                self.state[index+self.stock_dim+1] -= min(abs(action), self.state[index+self.stock_dim+1])
                # 手数料を計算
                self.cost +=self.state[index+1]*min(abs(action),self.state[index+self.stock_dim+1]) * \
                self.transaction_cost_pct
                self.trades+=1
            else:
                pass


    
    def _buy_stock(self, index, action):
        # sellの逆を行っている
        def _do_buy():
            available_amount = self.state[0] // self.state[index+1]
            # print('available_amount:{}'.format(available_amount))
            
            #update balance
            self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                              (1+ self.transaction_cost_pct)

            self.state[index+self.stock_dim+1] += min(available_amount, action)
            
            self.cost+=self.state[index+1]*min(available_amount, action)* \
                              self.transaction_cost_pct
            self.trades+=1
        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            _do_buy()
        else:
            # turbulence_indexを用いる場合は閾値以下である必要がある
            if self.turbulence< self.turbulence_threshold:
                _do_buy()
            else:
                pass

    def _make_plot(self):
        # 評価額の推移を画像として保存
        plt.plot(self.asset_memory,'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()


    def step(self, actions):
        # 取引期間の最終日かどうか
        self.terminal = self.day >= len(self.df.index.unique())-1
        
        # 最終日だったら （取引は行わない）
        if self.terminal:
            # print(f"Episode: {self.episode}")
            # make_plotsがTrueだったら
            if self.make_plots:
                # results/に評価額の推移の画像を保存
                self._make_plot()
            # 最終的な資産
            end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            # 資産の推移をDFに変換
            df_total_value = pd.DataFrame(self.asset_memory)
            # 初期資産を現在の資産から引くことでトータルでの利益を計算
            tot_reward = self.state[0]+sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))- self.initial_amount 
            # カラム名を設定
            df_total_value.columns = ['account_value']
            # 階差を計算することで一日ごとの利益を計算
            df_total_value['daily_return']=df_total_value.pct_change(1)
            # 一日ごとの利益の標準偏差が0じゃなかったらシャープレシオを計算 保存用ではなくログ用っぽい
            if df_total_value['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
            # 報酬の推移をDFに変換 ↑のdaily_returnと何がちがうんだろう
            df_rewards = pd.DataFrame(self.rewards_memory)
            # エピソード数がprint_verbosityの倍数になるごとにログを出力
            if self.episode%self.print_verbosity ==0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset:{self.asset_memory[0]:0.2f}")           
                print(f"end_total_asset:{end_total_asset:0.2f}")
                print(f"total_reward:{tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value['daily_return'].std() !=0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")
            return self.state, self.reward, self.terminal,{}

        # 最終日以外
        else:
            # 正規化されていたactionを売買する株式数に変更
            actions = actions * self.hmax
            # 各ステップのactionを記録
            self.actions_memory.append(actions)
            #actions = (actions.astype(int))
            # turbulence_indexの閾値が設定されていたら
            if self.turbulence_threshold is not None:
                # 閾値を超えていたら
                if self.turbulence>=self.turbulence_threshold:
                    # すべての株式を-hmax株売却
                    # あとでどうせ全部売る処理をするのにこのif文になんの意味があるんだろう 記録用?
                    actions=np.array([-self.hmax]*self.stock_dim)
            #　取引前の資産 残りの現金　+　各株式の株価＊保有株式数
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            # 小さい順に並べて元のインデックスを取得 [3,1,2] → [1, 2, 0]
            argsort_actions = np.argsort(actions)
            # np.where(actions<0)[0].shape[0]は売る選択をした銘柄の数を表している
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            # 逆の配列に対して↑と同様の計算を行えば買う選択をした銘柄のインデックスが求まる
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            # 売る
            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            # 買う
            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            # 日付を一日進める
            self.day += 1
            # 次の日の株価、テクニカル指標などを取得
            self.data = self.df.loc[self.day,:]    
            # turbulence_indexを使用している場合はこれを更新
            if self.turbulence_threshold is not None:     
                self.turbulence = self.data['turbulence'].values[0]
            # 状態を更新
            self.state =  self._update_state()
            # 更新後の資産 残りの現金　+　各株式の株価＊保有株式数
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            # 現在の資産を記録
            self.asset_memory.append(end_total_asset)
            # 日付を記録
            self.date_memory.append(self._get_date())
            # 取引後の資産 - 取引前の資産 = 報酬
            self.reward = end_total_asset - begin_total_asset
            # 報酬を記録
            self.rewards_memory.append(self.reward)
            # 報酬をスケーリング　学習の安定化のため？
            self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        #self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        #initiate state
        self.state = self._initiate_state()
        self.episode+=1
        return self.state
    
    def render(self, mode='human',close=False):
        return self.state

    def _initiate_state(self):
        # singleとmultipleの両方で状態を生成（[balance, 株価, 持ち株数, 各テクニカル指標]を生成）
        # sumはlistの連結を行っている
        if len(self.df.tic.unique())>1:
            # for multiple stock
            state = [self.initial_amount] + \
                     self.data.close.values.tolist() + \
                     [0]*self.stock_dim  + \
                     sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
        else:
            # for single stock
            state = [self.initial_amount] + \
                    [self.data.close] + \
                    [0]*self.stock_dim  + \
                    sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        return state

    def _update_state(self):
        # 基本的な流れは_initiate_stateと同じ
        # self.dataの更新を行った後に呼ばれる必要がある
        # 保有株式数のstateの更新はここではなく_sell_stock, _buy_stockで行われる
        if len(self.df.tic.unique())>1:
            # for multiple stock
            state =  [self.state[0]] + \
                      self.data.close.values.tolist() + \
                      list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])

        else:
            # for single stock
            state =  [self.state[0]] + \
                     [self.data.close] + \
                     list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                     sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
                          
        return state

    def _get_date(self):
        if len(self.df.tic.unique())>1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_asset_memory(self):
        # saveってなってるけど保存する処理はなくて資産の推移のDataFrameを日付付きで返している
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique())>1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
