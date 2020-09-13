# coding=utf-8
import matplotlib.pyplot as plt
import sys


# 读取output log
def log_reader(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    n_epoch = (len(lines) - 2) // 3
    print('Totally %d epochs' % n_epoch)
    Train_Cost, Valid_Cost, Test_Cost, Train_Acc, Valid_Acc, Test_Acc = [[] for i in range(6)]
    for i in range(n_epoch):
        # print(i)
        cost_val = lines[i * 3].strip()
        acc_val = lines[i * 3 + 1].strip()

        cost_val = cost_val.split(',')
        Train_Cost.append(float(cost_val[1].split(':')[1]))
        Valid_Cost.append(float(cost_val[2].split(':')[1]))
        Test_Cost.append(float(cost_val[3].split(':')[1]))

        acc_val = acc_val.split(',')
        Train_Acc.append(float(acc_val[0].split(':')[1]))
        Valid_Acc.append(float(acc_val[1].split(':')[1]))
        Test_Acc.append(float(acc_val[2].split(':')[1]))

    # 检查是否读取完全
    assert len(Train_Cost) == len(Train_Acc) == n_epoch

    return Train_Cost, Valid_Cost, Test_Cost, Train_Acc, Valid_Acc, Test_Acc


def train_visualization(output_path):
    """visualize the training process
    """    
    log_path = output_path + 'output.log'
    Train_Cost, Valid_Cost, Test_Cost, Train_Acc, Valid_Acc, Test_Acc = log_reader(log_path)
    n_epoch = len(Train_Cost)

    x1 = range(n_epoch)
    x2 = range(n_epoch)
    y1 = Train_Cost
    y2 = Valid_Cost
    y3 = Test_Cost
    y4 = Train_Acc
    y5 = Valid_Acc
    y6 = Test_Acc
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, label="Train_Cost", linewidth=2)
    plt.plot(x1, y2, label="Valid_Cost", linewidth=2)
    plt.plot(x1, y3, label="Test_Cost", linewidth=2)

    plt.title('binary cross entropy vs. epoches')
    plt.ylabel('binary cross entropy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y4, label="Train_Acc", linewidth=2)
    plt.plot(x2, y5, label="Valid_Acc", linewidth=2)
    plt.plot(x2, y6, label="Test_Acc", linewidth=2)
    plt.xlabel('Accuracy@20 vs. epoches')
    plt.ylabel('Accuracy@20')
    plt.legend(loc='best')
    plt.savefig(output_path + 'loss_fig.png')
    # plt.show()


def compare_outputs(output_paths):
    """Compare several training processes between experiments
    """
    cost_list = []
    acc_list = []
    for output_path in output_paths:
        _, valid_cost, _, _, valid_acc, _ = log_reader(output_path + 'output.log')
        cat_name = output_path.split('/')[-2]
        cost_list.append((valid_cost, cat_name))
        acc_list.append((valid_acc, cat_name))

    n_epoch = len(cost_list[0][0])
    x1 = range(n_epoch)
    x2 = range(n_epoch)
    plt.subplot(2, 1, 1)
    for item in cost_list:
        plt.plot(x1, item[0], label=item[1], linewidth=2)
    plt.title('binary cross entropy vs. epoches')
    plt.ylabel('binary cross entropy')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    for item in acc_list:
        plt.plot(x2, item[0], label=item[1], linewidth=2)
    plt.xlabel('Accuracy@20 vs. epoches')
    plt.ylabel('Accuracy@20')
    plt.legend(loc='best')

    plt.savefig('cmp_between' + str([item[1] for item in acc_list]) + '.png')


if __name__ == '__main__':
    args = sys.argv
    if len(args) <= 1:
        print('Please input the output path!')
        exit(0)
    elif len(args) == 2:
        print('Drawing the loss fig')
        path = args[1]
        train_visualization(path)
    else:
        print('Drawing the comparing fig')
        paths = args[1:]
        compare_outputs(paths)
