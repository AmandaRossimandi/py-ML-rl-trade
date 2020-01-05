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
def run_dqn(stock_name, num_features, num_neurons, episodes, batch_size, random_action_decay, future_reward_importance):

    # returns a list of stocks closing price
    df = pd.read_csv(INPUT_CSV_TEMPLATE % stock_name)
    data = df['Close'].astype(float).tolist()  #https://www.kaggle.com/camnugent/sandp500
    l = len(data) - 1

    print(f'Running {episodes} episodes, on {stock_name} (has {l} rows), features={num_features}, '
          f'batch={batch_size}, random_action_decay={random_action_decay}')
    dqn = Dqn()
    profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name, num_trains, eps = \
        dqn.learn(data, episodes, num_features, batch_size, USE_EXISTING_MODEL, RANDOM_ACTION_MIN,
                  random_action_decay, num_neurons, future_reward_importance)

    print(f'Learning completed. Backtest the model {model_name} on any stock')
    print('python backtest.py ')

    print(f'see plot of profit_vs_episode = {profit_vs_episode[:10]}')
    plot_barchart(profit_vs_episode, "episode vs profit", "episode vs profit", "total profit", "episode", 'green')

    print(f'see plot of trades_vs_episode = {trades_vs_episode[:10]}')
    plot_barchart(trades_vs_episode, "episode vs trades", "episode vs trades", "total trades", "episode", 'blue')

    text = f'{stock_name} ({l}), features={num_features}, nn={num_neurons},batch={batch_size}, ' \
           f'epi={episodes}({num_trains}), eps={np.round(eps, 1)}({np.round(random_action_decay, 5)})'
    print(f'see plot of epsilon_vs_episode = {epsilon_vs_episode[:10]}')
    plot_barchart(epsilon_vs_episode	,  "episode vs epsilon", "episode vs epsilon",
                  "epsilon(probability of random action)", text, 'red')
    print(text)


if __name__ == "__main__":
    # Running example:
    # python rl_dqn.py --stock_name '^GSPC_01' --num_features 1 --num_neurons 4 --episodes 1800 --batch_size 1

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_name'              , '-s' , type=str, default='test_sinus')  # ^GSPC_2001_2010  ^GSPC_1970_2018  ^GSPC_2011
    parser.add_argument('--num_episodes'            , '-ne', type=int, default=20)  # (int) > 0 ,minimum 200 episodes for results. episode represent trade and learn on all data.
    parser.add_argument('--num_features'            , '-nf', type=int, default=20)  # (int) > 0
    parser.add_argument('--num_neurons'             , '-nn', type=int, default=64)  # # (int) > 0
    parser.add_argument('--n_batch_size'            , '-nb', type=int, default=20)  # (int) > 0 size of a batched sampled from replay buffer for training
    parser.add_argument('--random_action_decay'            , type=float, default=0.99999)  # (float) 0-1
    parser.add_argument('--future_reward_importance'       , type=float, default=0.05)  # (float) 0-1 aka decay or discount rate, determines the importance of future rewards
    # If=0 agent will only learn to consider current rewards. if=1 it will make it strive for a long-term high reward.
    #python rl_dqn.py -s 'test_sinus' -nf 20 -nn 64 -e 2 -bs 20

    args = parser.parse_args()
    stock_name               = args.stock_name
    num_features             = args.num_features
    num_neurons              = args.num_neurons
    num_episodes             = args.episodes
    n_batch_size             = args.batch_size
    random_action_decay      = args.random_action_decay
    future_reward_importance = args.future_reward_importance

    run_dqn(stock_name=stock_name, num_features=num_features, num_neurons=num_neurons, episodes=num_episodes,
            batch_size=n_batch_size, random_action_decay=random_action_decay,
            future_reward_importance=future_reward_importance)
