import numpy as np
import pandas as pd
from dqn import Dqn
from utils import plot_barchart, record_run_time
from consts import INPUT_CSV_TEMPLATE


RANDOM_ACTION_MIN = 0.05    # (float)
USE_EXISTING_MODEL = False  # (bool) do not touch this

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)  # prevent numpy exponential notation on print, default False


@record_run_time
def run_dqn(name_asset, n_features, n_neurons, n_episodes, batch_size, random_action_decay, future_reward_importance):

    # returns a list of stocks closing price
    df = pd.read_csv(INPUT_CSV_TEMPLATE % name_asset)
    data = df['Close'].astype(float).tolist()  #https://www.kaggle.com/camnugent/sandp500
    l = len(data) - 1

    print(f'Running {n_episodes} episodes, on {name_asset} (has {l} rows), features={n_features}, '
          f'batch={batch_size}, random_action_decay={random_action_decay}')
    dqn = Dqn()
    profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name, num_trains, eps = \
        dqn.learn(data, n_episodes, n_features, batch_size, USE_EXISTING_MODEL, RANDOM_ACTION_MIN,
                  random_action_decay, n_neurons, future_reward_importance)

    print(f'Learning completed. Backtest the model {model_name} on any stock')
    print('python backtest.py ')

    print(f'see plot of profit_vs_episode = {profit_vs_episode[:10]}')
    plot_barchart(profit_vs_episode, "episode vs profit", "episode vs profit", "total profit", "episode", 'green')

    print(f'see plot of trades_vs_episode = {trades_vs_episode[:10]}')
    plot_barchart(trades_vs_episode, "episode vs trades", "episode vs trades", "total trades", "episode", 'blue')

    text = f'{name_asset} ({l}), features={n_features}, nn={n_neurons},batch={batch_size}, ' \
           f'epi={n_episodes}({num_trains}), eps={np.round(eps, 1)}({np.round(random_action_decay, 5)})'
    print(f'see plot of epsilon_vs_episode = {epsilon_vs_episode[:10]}')
    plot_barchart(epsilon_vs_episode	,  "episode vs epsilon", "episode vs epsilon",
                  "epsilon(probability of random action)", text, 'red')
    print(text)


if __name__ == "__main__":
    # Running example:
    # python rl_dqn.py -na 'test_sinus' -ne 2000 -nf 20 -nn 64 -nb 20 

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_asset'              , '-na', type=str, default='test_sinus')  # ^GSPC_2001_2010  ^GSPC_1970_2018  ^GSPC_2011    test_sinus
    parser.add_argument('--num_episodes'            , '-ne', type=int, default=20)  # (int) > 0 ,minimum 20,000 episodes for good results. episode represent trade and learn on all data.
    parser.add_argument('--num_features'            , '-nf', type=int, default=20)  # (int) > 0
    parser.add_argument('--num_neurons'             , '-nn', type=int, default=64)  # (int) > 0
    parser.add_argument('--n_batch_size'            , '-nb', type=int, default=20)  # (int) > 0 size of a batched sampled from replay buffer for training
    parser.add_argument('--random_action_decay'            , type=float, default=0.99999)  # (float) 0-1
    parser.add_argument('--future_reward_importance'       , type=float, default=0.05)  # (float) 0-1 aka decay or discount rate, determines the importance of future rewards
    # If=0 agent will only learn to consider current rewards. if=1 it will make it strive for a long-term high reward.
    #python rl_dqn.py -na 'test_sinus' -nf 20 -nn 64 -ne 2 -bs 20

    args = parser.parse_args()
    name_asset               = args.name_asset
    num_features             = args.num_features
    num_neurons              = args.num_neurons
    num_episodes             = args.num_episodes
    n_batch_size             = args.n_batch_size
    random_action_decay      = args.random_action_decay
    future_reward_importance = args.future_reward_importance

    run_dqn(name_asset=name_asset, n_features=num_features, n_neurons=num_neurons, n_episodes=num_episodes,
            batch_size=n_batch_size, random_action_decay=random_action_decay,
            future_reward_importance=future_reward_importance)
