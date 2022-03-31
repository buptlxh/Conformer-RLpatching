import argparse
import math

import numpy as np
import torch

import exp.exp_informer as EI
from utils.metrics import metric,MSE,RMSE

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=False, default='Informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=False, default='renewable_gen_p_max', help='data')
parser.add_argument('--root_path', type=str, default='/root/lixinhang/Conformer/data/renewable_gen_p_max/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='renewable_gen_p_max.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='/root/lixinhang/Conformer/checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=56, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=24, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=15, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=18, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=18, help='decoder input size')
parser.add_argument('--c_out', type=int, default=18, help='output size')

parser.add_argument('--d_model', type=int, default=2048, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=128, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='15,14,13,12,11,10,9,8,7,6,5,4,3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=4096, help='dimension of fcn')

parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=500, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
#args.use_gpu =False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {

    'renewable_gen_p_max':{'data':'renewable_gen_p_max.csv','T':'OT','M':[18,18,18],'S':[1,1,1],'MS':[18,18,1]}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
                args.embed, args.distil, args.mix, args.des, 0)

class Conformer():
    def __init__(self,start_index):
        self.storage_pool_predict=[]
        self.storage_pool_true = []
        self.exp=EI.Exp_Informer(args)
        self.start=start_index
        self.out_length = 10

    def pre(self,t):
        t_current = t - self.start
        preds, trues = self.exp.predict(setting, t,load=True)
        self.store_pred(preds)
        self.store_true(trues)
        true_tensteps=[]
        for i in range(self.out_length):
            true_tensteps.append(trues[i])
        if t_current < 5:
            pred_final = []
            for i in range(self.out_length):
                pred_final.append(preds[i])
            return pred_final
        else:
            pred_former = []
            true_former = []
            for i in range(5):
                idx = len(self.storage_pool_predict)-2-i
                pred_former.append(self.storage_pool_predict[idx])
                true_former.append(self.storage_pool_true[idx])


            con_weights = self.cal_confidence(pred_former, true_former)
            pred_final = self.cal_pred_final(pred_former, con_weights,true_tensteps)

            print("rmse", RMSE(np.array(true_tensteps), np.array(pred_final)))
            return pred_final




    def cal_confidence(self,pred_former,true_former):
        m_list = []
        n_list = []

        for i in range(len(pred_former)):
            m = 0
            n = 0
            for j in range(i+1):
                mae, mse, rmse, mape, mspe = metric(pred_former[i][j], true_former[i][j])

                if rmse <= 5:
                    m = m+1
                else:
                    n = n+1
            m_list.append(m)
            n_list.append(n)

        cr_list = []
        for i in range(len(pred_former)):
            cr = m_list[i]/(m_list[i]+n_list[i])
            cr_list.append(cr)
        cr_sum = 0
        for i in range(len(cr_list)):
            cr_sum = cr_sum + cr_list[i]

        cred_weights = []
        for i in range(len(pred_former)):
            cw = cr_list[i]/cr_sum
            cred_weights.append(cw)

        return cred_weights

    def cal_pred_final(self, pred_former, cred_weights,true_tensteps):
        pred_select = []
        for i in range(len(pred_former)):
            pred_select_each = []
            for j in range(self.out_length):
                pred_select_each.append(pred_former[i][i+j+1])

            pred_select.append(pred_select_each)

        pred_select_weight = np.zeros((5,self.out_length,18))
        for i in range(len(pred_select)):
            for j in range(len(pred_select[i])):
                for k in range(len(pred_select[i][j])):
                    pred_select_weight[i][j][k] = pred_select[i][j][k]*cred_weights[i]


        pred_final=np.zeros((self.out_length,18))
        for i in range(len(pred_select)):
            pred_final=pred_final+np.array(pred_select_weight[i])
        return pred_final


    def store_pred(self, preds):
        self.storage_pool_predict.append(preds)

    def store_true(self, trues):
        self.storage_pool_true.append(trues)

