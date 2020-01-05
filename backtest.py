from keras.models import load_model
from ai_agent import Agent
from dqn import Dqn
from utils import *
import argparse


def bt(data, num_features, use_existing_model, model_name):
    dqn = Dqn()
    dqn.open_orders = [data[0]]
    agent = Agent(num_features, use_existing_model, model_name)
    state = dqn.get_state(data, num_features, num_features)
    total_profits = 0
    total_holds = 0
    total_buys = 1
    total_sells = 0
    total_notvalid = 0
    l = len(data) - 1

    for t in range(num_features, l):

        action = agent.choose_best_action(state)  # it will always predict

        reward, total_profits, total_holds, total_buys, total_sells, total_notvalid = \
            dqn.execute_action(action, data[t], t, total_profits, total_holds, total_buys, total_sells, total_notvalid)

        done = True if t == l - 1 else False

        next_state = dqn.get_state(data, t + 1, num_features)
        #print(f'row #{t} {agent.actions[action]} @{data[t]}, state1={state}, state2={next_state}, reward={reward}')
        state = next_state

        if done:
            # sell position at end of episode
            reward, total_profits, total_holds, total_buys, total_sells, total_notvalid = \
                dqn.execute_action(2, data[t+1], t+1, total_profits, total_holds, total_buys, total_sells,
                                   total_notvalid)
            print("-----------------------------------------")
            print(f'Total Profit: {formatPrice(total_profits*100)} ,'
                  f' Total hold/buy/sell/notvalid trades: {total_holds} / {total_buys} / {total_sells} / {total_notvalid}')
            print("-----------------------------------------")


def main():
    """
    Example usage:
    python backtest.py --model_name model_ep180 --stock_name '^GSPC_1970_2018' --trading_fee 0.1

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_name', '-s', type=str, default='^GSPC_2011') #^GSPC_2011  GSPC_2019 GSPC_1970_2019 GSPC_1970_2018
    parser.add_argument('--model_name', '-mn', type=str, default='model_ep20000') # #model_ep0, model_ep10, model_ep20, model_ep30
    parser.add_argument('--trading_fee', '-tf', type=float, default=0.)
    #python backtest.py -s 'test_sinus' -mn 'model_ep2' -tf 0.
    args = parser.parse_args()
    stock_name = args.stock_name
    model_name = args.model_name
    trading_fee = args.trading_fee

    model_inst = load_model("files/output/" + model_name)
    num_features = model_inst.layers[0].input.shape.as_list()[1]
    use_existing_model = True
    data = getStockDataVec(stock_name)
    l = len(data) - 1

    print(f'starting back-testing model {model_name} on {stock_name} (file has {l} rows), features = {num_features} ')
    bt(data, num_features, use_existing_model, model_name)
    print(f'finished back-testing model {model_name} on {stock_name} (file has {l} rows), features = {num_features} ')


if __name__ == "__main__":
    main()
