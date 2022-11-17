import os

# show best runs from train output

FILES = [f'my_logs/{fl}' for fl in os.listdir('my_logs') if 's2' in fl]

if __name__ == '__main__':
    for file in FILES:
        with open(file, 'r') as inpt:
            lines = inpt.readlines()
            best_epoch = None
            best_acc = None
            epoch = None
            for line in lines:
                if 'epoch' in line:
                    epoch = int(line.split('epoch ')[-1])
                if 'Validation GP Accuracy::' in line:
                    acc = float(line.split(':: ')[-1])
                    if best_acc is None or acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch
        print(file, best_epoch, best_acc)