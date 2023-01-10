import os
import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch import optim
from models.EF_LSTM import EF_LSTM
from config.config import Config
from load_datas.load_data_pyt_new import MMdataloader
from dict_to_str import dict_to_str
# from metrics import Metrics
import pandas as pd
import random
from tqdm import tqdm
from AMIO import AMIO
device=torch.device("cpu")
# metrics = Metrics().metrics
criterion = nn.CrossEntropyLoss()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model,dataloader,config):
    def adjust_learning_rate(optimizer, epoch, config):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = config.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, verbose=True, patience=10)
    # initialize results
    best_acc = 0
    best_valid = 1e8
    sum_acc = 0
    epochs, best_epoch = 0, 0
    train_total_acc,valid_total_acc=[],[]
    train_total_loss, valid_total_loss=[],[]
    for s in range(500):
        # adjust_learning_rate(optimizer, epochs, config)
        epochs += 1
        # train
        y_pred, y_true = [], []
        losses = []
        model.train()
        train_loss = 0.0
        i = 1
        with tqdm(dataloader['train']) as td:
            for batchdata in td:
                audio = batchdata['audio'].to(device)
                text = batchdata['text'].to(device)
                labels = batchdata['label'].to(device)
                # clear gradient
                optimizer.zero_grad()
                # forward
                a_out = model(text, audio)
                sum = 0
                for name, param in model.named_parameters():
                    mul = 1
                    for size_ in param.shape:
                        mul *= size_
                    sum += mul
                    # print('%14s: %s' % (name, param.shape))
                # print(i,'参数个数:',sum)
                i = i + 1
                a_out = a_out.to(device)

                # a_out = a_out.view(-1)
                # print(a_out.size()) #768
                # a_out=a_out.unsqueeze(1)
                # a_out=np.argmax(a_out.detach().numpy(),axis=1)
                # print(torch.tensor(a_out,))
                # labels = Variable(labels)
                labels = np.argmax(labels.detach().numpy(), axis=1)
                # print(labels)
                labels = torch.tensor(labels, dtype=torch.float32)
                # compute_loss


                loss = criterion(a_out, labels.long())
                # backward
                loss.backward()
                # update
                optimizer.step()
                # store results
                train_loss += loss.item()
                y_pred.append(a_out)
                y_true.append(labels)
        train_loss = train_loss / len(dataloader['train'])
        print("TRAIN-'AUDIO' (%d/%d)>> loss: %.4f" % (epochs - best_epoch, epochs, train_loss))
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        all_true_label = true
        all_predicted_label = np.argmax(pred.detach().numpy(), axis=1)
        f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
        acc_score = accuracy_score(all_true_label, all_predicted_label)
        print("epoch {}  train_acc: {} f1:{}".format(epochs, acc_score, f1))

        # validation
        val_acc, val_f1, val_loss = test(model, dataloader['test'])
        print("epoch {},test_acc: {} f1:{}".format(epochs, val_acc, val_f1))
        valid_total_acc.append(val_acc)
        train_total_loss.append(train_loss)
        valid_total_loss.append(val_loss)
        # scheduler.step(val_loss)
        # if self.patience > 0:
        #     scheduler.step(val_acc)
        # save best model:

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epochs
            best_valid = val_loss
            print('best_epoch', best_epoch)
            model_path = os.path.join('results/model_save\\', 'eflstm.pth'.format(best_epoch))
            if os.path.exists(model_path):
                os.remove(model_path)
            # save model
            torch.save(model.state_dict(), model_path)
            # plt.subplot(1, 2, 1)
            # x1=range(1,epochs+1)
            # plt.plot(x1, train_total_acc, color='red', marker='o', label="Train_Acc")
            # plt.plot(x1, valid_total_acc, color='blue', marker='*', label="Valid_Acc")
            # plt.title("Train acc vs Valid acc")
            # plt.legend(loc='best')
            # plt.subplot(1, 2, 2)
            # x2 = range(1, epochs + 1)
            # plt.plot(x2, train_total_loss, color='red', marker='o', label="Train_Loss")
            # plt.plot(x2, valid_total_loss, color='blue', marker='*', label="Valid_Loss")
            # plt.title("Train loss vs Valid loss")
            # plt.legend(loc='best')
            # plt.show()
            # return


def test(model,dataloader):
    model.eval()  # 不加的话，测试的时候有输入数据，即使不训练，它也会改变权值，有batchnormalize的性质
    y_pred, y_true = [], []
    eval_loss = 0.0
    i = 0
    with torch.no_grad():
        with tqdm(dataloader) as td:
            for batchdata in td:
                audio = batchdata['audio'].to(device)
                text = batchdata['text'].to(device)
                labels = batchdata['label'].to(device)
                a_out = model(text,audio)
                labels = np.argmax(labels.detach().numpy(), axis=1)
                # print(labels)
                labels = torch.tensor(labels, dtype=torch.float32)
                loss = criterion(a_out, labels.long())
                eval_loss += loss.item()
                y_pred.append(a_out)
                y_true.append(labels)
    eval_loss = eval_loss / len(dataloader)
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    all_true_label = true
    all_predicted_label = np.argmax(pred.detach().numpy(), axis=1)
    f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
    acc_score = accuracy_score(all_true_label, all_predicted_label)
    return acc_score, f1, eval_loss


def run(dataloader):
    if not os.path.exists('results/model_save'):
        os.makedirs('results/model_save')


    model = AMIO(config).to(device)
    # model = EF_LSTM(config).to(device)
    print("#" * 40)

    # do train
    train(model,dataloader,config)
    # load pretrained model
    pretrained_path = os.path.join('./results/model_save\\', 'eflstm.pth')
    assert os.path.exists(pretrained_path)
    model.load_state_dict(torch.load(pretrained_path))
    # do test
    # using valid dataset to debug hyper parameters
    acc, f1, t = test(model,dataloader['test'])

    # plt.figure(figsize=(30, 20))
    # sns.heatmap(t, vmax=100, vmin=0)
    # plt.show()

    print('acc', acc, 'f1', f1)
if __name__ == '__main__':
    # pkl_file = open('data.pkl', 'rb')
    # data = pickle.load(pkl_file)
    # audio_train=data['audio_train']
    # text_train=data['text_train']
    # audio_test=data['audio_test']
    # text_test=data['text_test']
    # y_train=data['y_train']
    # y_test=data['y_test']
    # pkl_file.close()
    config = Config()
    dataloader=MMdataloader(config)
    # x_train=torch.cat((torch.tensor(text_train, dtype=torch.float32),torch.tensor(audio_train, dtype=torch.float32)),dim=1)
    # x_test=torch.cat((torch.tensor(text_test, dtype=torch.float32),torch.tensor(audio_test, dtype=torch.float32)),dim=1)
    # train_dataset = Data.TensorDataset(torch.tensor(x_train, dtype=torch.float32),
    #                                    torch.tensor(y_train, dtype=torch.float32))
    # train_loader = Data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)
    # test_dataset = Data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
    #                                   torch.tensor(y_test, dtype=torch.float32))
    # test_loader = Data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)


    seed = 1
    setup_seed(seed)
    run(dataloader)
    # run_debug(seeds,debug_times=50)




