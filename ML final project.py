import matplotlib.pyplot as pyplot
import numpy as np
import math
import time
import csv
import pandas as pd

import torch
import torchvision.transforms as transforms
import torch.utils.data as data_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


datapath = "/Users/jessica.w/Downloads/regressionDatasets/"

datalist = ["Automobile.csv","sleep.csv","servo.csv","pwLinear.csv",
            "analcatdata_runshoes.csv","cps_85_wages.csv","boston.csv","house_prices_kaggle.csv",
            "cholesterol.csv","pyrim.csv","iq_brain_size.csv","ERA.csv","quake.csv","happinessRank_2015.csv",
            "forest-fires.csv","nasa_numeric.csv","cpu_with_vendor.csv","strikes.csv","autoHorse.csv",
            "pbc.csv"]

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, (inputs, target) in enumerate(loader):
        inputs = inputs.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)
        #loss = torch.nn.MSELoss(reduction = "sum")
        
        #l = loss(model.layer(input_var).reshape(target_var),target_var)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        #loss_sum += loss.data[0] * inputs.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()
    return {'loss':l,'accuracy': correct / len(loader.dataset) * 100.0}

    #return {'loss': loss_sum / len(loader.dataset),'accuracy': correct / len(loader.dataset) * 100.0,}
    


def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, (inputs, target) in enumerate(loader):
        inputs = inputs.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        #loss = criterion(output, target_var)
        loss = torch.nn.MSELoss(reduction = "sum")

        l = loss(output,target_var)
        #loss_sum += loss.data[0] * inputs.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()
    return {'loss':l,'accuracy': correct / len(loader.dataset) * 100.0,}
    #return {'loss': loss_sum / len(loader.dataset),'accuracy': correct / len(loader.dataset) * 100.0,}


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for inputs, _ in loader:
        inputs = inputs.cuda()
        input_var = torch.autograd.Variable(inputs)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def get_data(data_set,data_path,batch_size, num_workers):

    data = pd.read_csv(datapath+data_set)

    data = data.dropna()
    

    df_str_columns = data.select_dtypes(include= object)
    if len(df_str_columns.columns):
        data = pd.get_dummies(data,columns = df_str_columns.columns)
    data= data.astype('float')
    values = data.values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(values)

    X = scaled[:,0:-1]
    y = scaled[:,-1]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
    return X_train,X_test,y_train,y_test

def data_preprocess(X_train,X_test,y_train,y_test):
    

    X_train = torch.Tensor(list(X_train))
    Y_train = torch.Tensor(list(y_train))

    #X_train = X_train.type(torch.LongTensor)
    #Y_train = Y_train.type(torch.LongTensor)
    

    in_dim = X_train.shape[1]
    out_dim = y_train.shape[0]

    X_test = torch.Tensor(list(X_test))
    Y_test = torch.Tensor(list(y_test))

    train_set = data_utils.TensorDataset(X_train,Y_train)
    test_set = data_utils.TensorDataset(X_test,Y_test)
    val_set = data_utils.TensorDataset(X_train,Y_train)

    train_sampler = None
    val_sampler = None
    
    loaders = {'train': torch.utils.data.DataLoader(
        train_set,
        batch_size = batch_size,
        shuffle = True,
        sampler = train_sampler,
        num_workers = num_workers,
        pin_memory = True
        ),
        'val' : torch.utils.data.DataLoader(
            train_set,
            batch_size = batch_size,
            sampler = val_sampler,
            num_workers = num_workers,
            pin_memory = True
        ),
        'test' : torch.utils.data.DataLoader(
            test_set,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True
            )
                                                    
        }

    return loaders,in_dim,out_dim
def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor

class Net(torch.nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Net,self).__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(in_dim,n_hidden_1),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_hidden_1,n_hidden_2),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_hidden_2,out_dim)
            )
    def forward(self,x):
        x = self.layer(x)
        return x

class SequenceModel:
    base = Net
    args = list()
    kwargs = {'depth' : 1}


def main():
    parser = argparse.ArgumentParser(description='SGD/SWA training')
    parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
    parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
    parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
    parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
    parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
    parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
    parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print('Using model %s' % args.model)
    model_cfg = getattr(models, args.model)
    print('Loading dataset %s from %s' % (args.dataset, args.data_path))
    path = os.path.join(args.data_path, args.dataset.lower())
    X_train,X_test,y_train,y_test = get_data(datalist[0],datapath,args.batch_size,args.num_workers)
    loaders,in_dim,out_dim = data_preprocess(X_train,X_test,y_train,y_test)
    print('Preparing model')
    model = model_cfg.base(*model_cfg.args,in_dim = in_dim, n_hidden_1 = n_hidden_1, n_hidden_2=n_hidden_2,out_dim = out_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.swa:
        print('SWA training')
        swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes,**model_cfg.kwargs)
        swa_model.cuda()
        swa_n = 0
    else:
        print('SGD training')

    criterion = F.MSELoss
    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd)

    start_epoch = 0
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    if args.swa:
        swa_state_dict = checkpoint['swa_state_dict']
        if swa_state_dict is not None:
            swa_model.load_state_dict(swa_state_dict)
        swa_n_ckpt = checkpoint['swa_n']
        if swa_n_ckpt is not None:
            swa_n = swa_n_ckpt
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time'}
    if args.swa:
        columns = columns[:-1] + ['swa_te_loss', 'swa_te_acc'] + columns[-1:]
        swa_res = {'loss': None, 'accuracy': None}
    utils.save_checkpoint(
        args.dir,
        start_epoch,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        swa_n=swa_n if args.swa else None,
        optimizer=optimizer.state_dict())
    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
        train_res = train_epoch(loaders['train'], model, criterion, optimizer)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            test_res = utils.eval(loaders['test'], model, criterion)
        else:
            test_res = {'loss': None, 'accuracy': None}
        if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
            utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
                utils.bn_update(loaders['train'], swa_model)
                swa_res = utils.eval(loaders['test'], swa_model, criterion)
            else:
                swa_res = {'loss': None, 'accuracy': None}
        if (epoch + 1) % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                state_dict=model.state_dict(),
                swa_state_dict=swa_model.state_dict() if args.swa else None,
                swa_n=swa_n if args.swa else None,
                optimizer=optimizer.state_dict())
        time_ep = time.time() - time_ep
        values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep]
        if args.swa:
            values = values[:-1] + [swa_res['loss'], swa_res['accuracy']] + values[-1:]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
            print(table)
    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict() if args.swa else None,
            swa_n=swa_n if args.swa else None,
            optimizer=optimizer.state_dict())



    
    
