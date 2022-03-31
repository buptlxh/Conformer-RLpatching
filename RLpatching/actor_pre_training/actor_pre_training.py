import csv
import random
import torch.utils.data as Data
import numpy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import  OrderedDict
from torch import optim
from torch.autograd import Variable
import os
import sys
sys.path.append(r'~/lixinhang/RLpatching/')
BATCH_SIZE=256
LR = 0.001
epoch_num = 100000000000000


class Agent_Supervised_Actor(torch.nn.Module):
    def __init__(self):
        super(Agent_Supervised_Actor,self).__init__()

        self.fc_net = nn.Sequential(
            OrderedDict([
          ('dense1', nn.Linear(1897,1024)),
          ('norm1', nn.BatchNorm1d(1024)),
          ('relu1', nn.ELU()),
          ('dense2', nn.Linear(1024,512)),
          ('norm2', nn.BatchNorm1d(512)),
          ('relu2', nn.ELU()),
          ("dense3", nn.Linear(512, 256)),
          ("norm3", nn.BatchNorm1d(256)),
          ("relu3", nn.ELU()),
          ("dense4", nn.Linear(256, 128)),
          ("norm4", nn.BatchNorm1d(128)),
          ("relu4", nn.ELU()),
          ("dense5", nn.Linear(128,54))
        ])
        )

    def forward(self,x):
        if len(x.size())==3:
            x = torch.squeeze(x,1)

        return self.fc_net(x)


def str_list_to_float_list(str_list):
    n = 0
    while n < len(str_list):
        str_list[n] = float(str_list[n])
        n += 1
    return(str_list)


def dataset(train_data, format=2):
    train_data_size = 106820 * train_data
    train_data = []
    test_data=[]
    train_target = []
    test_target=[]

    train_data_a_ex =[]
    train_data_p_ex =[]
    train_data_v_ex =[]
    train_data_q_ex =[]
    train_data_p_or=[]
    train_data_v_or = []
    train_data_q_or = []
    train_data_rho =[]
    train_data_grid_loss =[]
    train_data_load_p =[]
    train_data_load_q =[]
    train_data_max_p =[]
    train_data_gen_q =[]

    test_data_a_ex =[]
    test_data_p_ex =[]
    test_data_v_ex =[]
    test_data_q_ex =[]
    test_data_p_or = []
    test_data_v_or = []
    test_data_q_or = []
    test_data_rho =[]
    test_data_grid_loss =[]
    test_data_load_p =[]
    test_data_load_q =[]
    test_data_max_p =[]
    test_data_gen_q =[]

    with open("data/a_ex.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_a_ex.append(row)
            elif i>train_data_size and i<=106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_a_ex.append(row)

    with open("data/p_ex.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_p_ex.append(row)
            elif i > train_data_size and i <= 106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_p_ex.append(row)

    with open("data/v_ex.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_v_ex.append(row)
            elif i > train_data_size and i <= 106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_v_ex.append(row)

    with open("data/q_ex.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_q_ex.append(row)
            elif i > train_data_size and i <= 106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_q_ex.append(row)

    with open("data/p_or.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_p_or.append(row)
            elif i > train_data_size and i <= 106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_p_or.append(row)

    with open("data/v_or.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_v_or.append(row)
            elif i > train_data_size and i <= 106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_v_or.append(row)

    with open("data/q_or.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_q_or.append(row)
            elif i > train_data_size and i <= 106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_q_or.append(row)

    with open("data/rho.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_rho.append(row)
            elif i>train_data_size and i<=106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_rho.append(row)

    with open("data/grid_loss.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_grid_loss.append(row)
            elif i>train_data_size and i<=106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_grid_loss.append(row)

    with open("data/load_p.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_load_p.append(row)
            elif i>train_data_size and i<=106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_load_p.append(row)

    with open("data/load_q.csv", "r", newline="") as csvfile2:
        reader2 = csv.reader(csvfile2)
        for i, rows in enumerate(reader2):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_load_q.append(row)
            elif i>train_data_size and i<=106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_load_q.append(row)

    with open("data/max_renewable_gen_p.csv", "r", newline="") as csvfile3:
        reader3 = csv.reader(csvfile3)
        for i, rows in enumerate(reader3):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_max_p.append(row)
            elif i>train_data_size and i<=106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_max_p.append(row)

    with open("data/gen_q.csv", "r", newline="") as csvfile4:
        reader4 = csv.reader(csvfile4)
        for i, rows in enumerate(reader4):
            if i > 0 and i <= train_data_size:
                row = rows
                row = str_list_to_float_list(row)
                train_data_gen_q.append(row)
            elif i>train_data_size and i<=106820:
                row = rows
                row = str_list_to_float_list(row)
                test_data_gen_q.append(row)
    # print("%%%%",np.array(test_data_load_p).shape)
    # print("%%%%",np.array(test_data_load_q).shape)
    # print("%%%%",np.array(test_data_max_p).shape
    # print("%%%%",np.array(test_data_gen_q).shape)
    if format == 1:
        row_y = []
        for i in range(10):
            row_y += [0 for _ in range(18)]
        for i in range(0,len(train_data_gen_q)-10):
            temp = train_data_a_ex[i] +train_data_p_ex[i] +train_data_q_ex[i] +train_data_v_ex[i] +train_data_p_or[i]+train_data_q_or[i]+train_data_v_or[i]+train_data_rho[i] +train_data_grid_loss[i]+train_data_load_p[i] + train_data_load_q[i]\
                + row_y + train_data_gen_q[i]
            train_data.append(temp)
        # print("***",np.array(train_data).shape)
        for i in range(0,len(test_data_gen_q)-10):
            temp = test_data_a_ex[i] +test_data_p_ex[i] +test_data_q_ex[i] +test_data_v_ex[i] +test_data_p_or[i]+test_data_q_or[i]+test_data_v_or[i]+test_data_rho[i] +test_data_grid_loss[i]+test_data_load_p[i] + test_data_load_q[i]\
                + row_y + test_data_gen_q[i]
            test_data.append(temp)
        # print("***",np.array(test_data).shape)
    elif format==0 or format==2:
        for i in range(0,len(train_data_gen_q)-10):
            temp = train_data_a_ex[i] +train_data_p_ex[i] +train_data_q_ex[i] +train_data_v_ex[i] +train_data_p_or[i]+train_data_q_or[i]+train_data_v_or[i]+train_data_rho[i] +train_data_grid_loss[i]+train_data_load_p[i] + train_data_load_q[i]\
                + train_data_max_p[i] + train_data_max_p[i+1] + train_data_max_p[i+2] + train_data_max_p[i+3] + train_data_max_p[i+4] + train_data_max_p[i+5] + train_data_max_p[i+6] + train_data_max_p[i+7] + train_data_max_p[i+8] + train_data_max_p[i+9] + train_data_gen_q[i]
            train_data.append(temp)
        # print("***",np.array(train_data).shape)
        for i in range(0,len(test_data_gen_q)-10):
            temp = test_data_a_ex[i] +test_data_p_ex[i] +test_data_q_ex[i] +test_data_v_ex[i] +test_data_p_or[i]+test_data_q_or[i]+test_data_v_or[i]+test_data_rho[i] +test_data_grid_loss[i]+test_data_load_p[i] + test_data_load_q[i]\
                + test_data_max_p[i] + test_data_max_p[i+1] + test_data_max_p[i+2] + test_data_max_p[i+3] + test_data_max_p[i+4] + test_data_max_p[i+5] + test_data_max_p[i+6] + test_data_max_p[i+7] + test_data_max_p[i+8] + test_data_max_p[i+9] + test_data_gen_q[i]
            test_data.append(temp)
        # print("***",np.array(test_data).shape)
    elif format==3:
        for i in range(0,len(train_data_gen_q)-10):
            row_y = []
            row_y += train_data_max_p[i]
            for i in range(9):
                row_y += [0 for _ in range(18)]
            temp = train_data_a_ex[i] +train_data_p_ex[i] +train_data_q_ex[i] +train_data_v_ex[i] +train_data_p_or[i]+train_data_q_or[i]+train_data_v_or[i]+train_data_rho[i] +train_data_grid_loss[i]+train_data_load_p[i] + train_data_load_q[i]\
                + row_y_train + train_data_gen_q[i]
            train_data.append(temp)
        # print("***",np.array(train_data).shape)
        for i in range(0,len(test_data_gen_q)-10):
            row_y = []
            row_y += test_data_max_p[i]
            for i in range(9):
                row_y += [0 for _ in range(18)]
            temp = test_data_a_ex[i] +test_data_p_ex[i] +test_data_q_ex[i] +test_data_v_ex[i] +test_data_p_or[i]+test_data_q_or[i]+test_data_v_or[i]+test_data_rho[i] +test_data_grid_loss[i]+test_data_load_p[i] + test_data_load_q[i]\
                + row_y_test + test_data_gen_q[i]
            test_data.append(temp)
        # print("***",np.array(test_data).shape)

    with open("data/gen_p.csv", "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if i <= train_data_size-10 and i > 0:
                row = rows
                row = str_list_to_float_list(row)
                train_target.append(row)

            elif i>train_data_size and i<=106810:
                row = rows
                row = str_list_to_float_list(row)
                test_target.append(row)
    # print('train_target:', np.array(train_target).shape)
    # print('test_target:', np.array(test_target).shape)
    return train_data,train_target,test_data,test_target


def fit(epoch,model,data_loader,optimizer,phase='training',volatile=False):
    if phase == "training":  # 判断当前是训练还是验证
        model.train()
    if phase == "validation":
        model.eval()
        volatile=True
    all_loss=0.0
    for batch_idx,(data_,target_) in enumerate(data_loader):
        # print('data[i]',data[i])
        data, target = data_.cuda(), target_.cuda()  # 使用cuda加速
        #target.argmax(1)
        #data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()  # 重置梯度
        output = model(data)  # 得出预测结果

        loss = torch.nn.functional.mse_loss(output, target)  # 计算损失值
        all_loss+=torch.nn.functional.mse_loss(output, target,size_average=False).item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    print("Training actor, epoch:",epoch,"----",phase,"loss:",all_loss/len(data_loader.dataset))

    return all_loss/len(data_loader.dataset)


def supervised_train_actor(model=None, phase='training', format=2, exp_name="pre_training_actor"):
   if model is not None:
      if os.path.exists('train_model_a_10.pth'):
         model = torch.load('train_model_a_10.pth')
         torch.save(model.state_dict(), "train_model_p_all_10.pth")
      else:
         model=Agent_Supervised_Actor()
         model.cuda(0)
   else:
      model.to("cuda")
   print("**********Reading the dataset************")
   train_data,train_target,test_data,test_target=dataset(1, format=format)

   train_dataset = Data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_target))

    # 把 dataset 放入 DataLoader
   train_loader = Data.DataLoader(
      dataset=train_dataset,  # torch TensorDataset format
      batch_size=BATCH_SIZE,  # mini batch size
      shuffle=True,  # 要不要打乱数据 (打乱比较好)
      num_workers=2,  # 多线程来读数据
   )

   optimizer = torch.optim.Adam(model.parameters(), lr=LR)
   StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
   train_losses, train_accuracy = [], []
   val_losses, val_accuracy = [], []

   with open(os.path.join('best_model', exp_name, "pre_training_actor_loss.csv"),"w",newline="") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["epoch","loss"])



   for epoch in range(0, epoch_num):
      epoch_loss = fit(epoch, model, train_loader, optimizer, phase='training')
      StepLR.step()

      if epoch %10==0:
         torch.save(model, os.path.join('best_model', exp_name, "train_model_a_10.pth"))
         torch.save(model.state_dict(), os.path.join('best_model', exp_name, "train_model_p_all_10.pth"))

      with open(os.path.join('best_model', exp_name, "pre_training_actor_loss.csv"),"a+",newline="") as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow([epoch,epoch_loss])

      train_losses.append(epoch_loss)
      print("current_lr: %s" % (optimizer.state_dict()['param_groups'][0]['lr']))
      if epoch > 1000 or epoch_loss <= 20:
         print("Complicate the training of actor!")
         return True
        

if __name__ == '__main__':
   result = supervised_train_actor()






