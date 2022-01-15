import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 75


def draw(config, iter_log, train_accs, test_accs):
    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(iter_log, train_accs, marker='o', markersize=3)
    plt.plot(iter_log, test_accs, marker='o', markersize=3)
    plt.legend(['Trainset Acc.', 'Testset Acc.'])
    save_filename = os.path.join(config['output_dir'], f"{config['experiment_name']}.png")
    plt.savefig(save_filename)
