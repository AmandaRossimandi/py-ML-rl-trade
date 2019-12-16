import numpy as np
from dqn import Dqn
from utils import seed, getStockDataVec, plot_barchart, record_run_time


##do not touch those params
RANDOM_ACTION_MIN   = 0.0  #(float) 0-1 do not touch this
USE_EXISTING_MODEL  = False#(bool)      do not touch this

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)  # prevent numpy exponential #notation on print, default False


@record_run_time
def run_dqn(stock_name, num_features, num_neurons, episodes, batch_size, random_action_decay, future_reward_importance):
    # episodes=2 +features=252 takes 6 minutes
    seed()
    data                = getStockDataVec(stock_name)#https://www.kaggle.com/camnugent/sandp500
    l                   = len(data) - 1

    print(f'Running {episodes} episodes, on {stock_name} (has {l} rows), features={num_features}, batch={batch_size}, random_action_decay={random_action_decay}')
    dqn=Dqn()
    profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name, num_trains , eps= \
        dqn.learn      (data, episodes, num_features, batch_size, USE_EXISTING_MODEL, RANDOM_ACTION_MIN, random_action_decay, num_neurons, future_reward_importance)

    print(f'i think i learned to trade. now u can backtest the model {model_name} on any stock')
    print('python backtest.py ')

    print(f'see plot of profit_vs_episode = {profit_vs_episode[:10]}')
    plot_barchart(profit_vs_episode	,  "episode vs profit", "episode vs profit", "total profit", "episode", 'green')

    print(f'see plot of trades_vs_episode = {trades_vs_episode[:10]}')
    plot_barchart(trades_vs_episode	,  "episode vs trades", "episode vs trades", "total trades", "episode", 'blue')

    text = f'{stock_name} ({l}), features={num_features}, nn={num_neurons},batch={batch_size}, epi={episodes}({num_trains}), eps={np.round(eps, 1)}({np.round(random_action_decay, 5)})'
    print(f'see plot of epsilon_vs_episode = {epsilon_vs_episode[:10]}')
    plot_barchart(epsilon_vs_episode	,  "episode vs epsilon", "episode vs epsilon", "epsilon(probability of random action)", text, 'red')
    print(text)


if __name__ == "__main__":
    # Running example:
    # python rl_dqn.py --stock_name '^GSPC_01' --num_features 1 --num_neurons 4 --episodes 1800 --batch_size 1

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_name', '-s', type=str, default='^GSPC_01')  # ^GSPC_2001_2010  ^GSPC_1970_2018  ^GSPC_2011
    parser.add_argument('--num_features', '-nf', type=int, default=1)  # (int) > 0   super simple features
    parser.add_argument('--num_neurons', '-nn', type=int, default=4)  # # (int) > 0
    parser.add_argument('--episodes', '-e', type=int, default=180)  # (int) > 0 ,minimum 200 episodes for results. episode represent trade and learn on all data.
    parser.add_argument('--batch_size', '-bs', type=int, default=1)  # (int) > 0 size of a batched sampled from replay buffer for training
    parser.add_argument('--random_action_decay', type=float, default=0.8993)  # (float) 0-1
    parser.add_argument('--future_reward_importance', type=float, default=0.95)  # (float) 0-1 aka decay or discount rate, determines the importance of future rewards.If=0 then agent will only learn to consider current rewards. if=1 it will make it strive for a long-term high reward.

    args = parser.parse_args()
    stock_name = args.stock_name
    num_features = args.num_features
    num_neurons = args.num_neurons
    episodes = args.episodes
    batch_size = args.batch_size
    random_action_decay = args.random_action_decay
    future_reward_importance = args.future_reward_importance

    run_dqn(stock_name=stock_name, num_features=num_features, num_neurons=num_neurons, episodes=episodes, batch_size=batch_size, random_action_decay=random_action_decay,
            future_reward_importance=future_reward_importance)

    #Total Profit:  %0.141 , Total trades: 158, hold_count: 0
    #0.95        0
    #0.9995     16
    #0.99995     0
    #0.999995   81
    #0.9999995   0
    #0.99999995  0
    #0.999999995 5