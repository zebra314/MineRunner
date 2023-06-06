import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import datetime


def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('Steve')
    plt.xlabel('epoch')
    plt.ylabel('rewards')


def DQN():
    DQN_Rewards = np.load("../../asset/Rewards/DQN_rewards.npy").transpose()
    # DQN_avg = np.mean(DQN_Rewards, axis=1)
    # DQN_std = np.std(DQN_Rewards, axis=1)

    initialize_plot()
    plt.plot(DQN_Rewards)
    # plt.plot([i for i in range(1000)], DQN_avg,
            #  label='DQN', color='blue')
    # plt.fill_between([i for i in range(1000)],
                    #  DQN_avg+DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    # plt.legend(loc="best")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filePath = f"../../asset/Plots/DQN_{current_time}.png"
    # 保存图形文件
    plt.savefig(filePath)
    plt.show()
    plt.close()

def CNN():
    CNN_Rewards = np.load("../../asset/Rewards/CNN_rewards_2023-06-06_19-53.npy").transpose()
    # DQN_avg = np.mean(DQN_Rewards, axis=1)
    # DQN_std = np.std(DQN_Rewards, axis=1)

    initialize_plot()
    plt.plot(CNN_Rewards)
    # plt.plot([i for i in range(1000)], DQN_avg,
            #  label='DQN', color='blue')
    # plt.fill_between([i for i in range(1000)],
                    #  DQN_avg+DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    # plt.legend(loc="best")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filePath = f"../../asset/Plots/CNN_{current_time}.png"
    # 保存图形文件
    plt.savefig(filePath)
    plt.show()
    plt.close()

if __name__ == "__main__":
    '''
    Plot the trend of Rewards
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--DQN", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--CNN", action="store_true")
    args = parser.parse_args()
    os.makedirs("../../asset/Plots", exist_ok=True)
    if args.DQN:
        DQN()
    elif args.CNN:
        CNN()

