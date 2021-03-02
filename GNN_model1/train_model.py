import math
import time
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import pearsonr

from GNN_model1.net import gtnet
from GNN_model1.optimization import Optim
from data_handler import DataLoaderS
from utils import rmse


def evaluate(data, XX, YY, model, evaluateL2, evaluateL1, args, return_oni_preds=False):
    model.eval()
    mask = args.mask
    num_target_nodes = np.count_nonzero(mask)
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    preds = None
    Ytrue = None

    i = 0
    for X, Y in data.get_batches(XX, YY, args.batch_size, shuffle=False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        Y = Y[:, mask]
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        output = output[:, mask]
        if preds is None:
            preds = output
            Ytrue = Y
        else:
            preds = torch.cat((preds, output))
            Ytrue = torch.cat((Ytrue, Y))
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * num_target_nodes)
        i += 1

    rse = math.sqrt(total_loss / n_samples)
    rae = (total_loss_l1 / n_samples)

    preds = preds.data.cpu().numpy()
    Ytest = Ytrue.data.cpu().numpy()
    sigma_p = preds.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = preds.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((preds - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    oni_Y = Ytest.mean(axis=1)  # mean over the ONI region, Ytest, preds have shape N_test x N_nodes
    oni_pred = preds.mean(axis=1)
    oni_corr = np.corrcoef(oni_Y, oni_pred)[0, 1]
    rmse_val = rmse(oni_Y, oni_pred)
    r, p = pearsonr(oni_Y, oni_pred)
    # print("{} index metrics: Correlation = {:.3f}, Mean MSE: {:.3f}, RMSE: {:.3f},"
    #      " r={:.3f}, p={:.3f}".format(args.index, oni_corr, mmse, rmse_val, r, p))
    oni_stats = {"Corrcoef": oni_corr, "RMSE": rmse_val, "Pearson_r": r, "Pearson_p": p}
    if return_oni_preds:
        return rse, rae, correlation, oni_stats, preds, Ytest
    else:
        return rse, rae, correlation, oni_stats


def train(data, XX, YY, model, criterion, optim, args):
    model.train()
    mask = torch.from_numpy(args.mask).to(args.device)  # eg mask out the relevant ONI/El Nino3.4 index region
    num_target_nodes = np.count_nonzero(args.mask)
    total_loss = 0
    n_samples = 0
    iter = 1
    for X, Y in data.get_batches(XX, YY, args.batch_size, shuffle=args.shuffle):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        if iter % args.step_size == 1:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.LongTensor(id).to(args.device)
            tx = X[:, :, id, :]
            ty = Y[:, id]
            preds = model(tx, id)  # shape = (batch_size x _ x #nodes x _)
            preds = torch.squeeze(preds)
            if not args.train_all_nodes:
                ty = ty[:, mask]
                preds = preds[:, mask] if len(preds.shape) > 1 else preds[mask]
            loss = criterion(preds, ty)
            loss.backward()
            total_loss += loss.item()
            n_samples += (preds.size(0) * num_target_nodes)
            grad_norm = optim.step()

        iter += 1
    return total_loss / iter


def main(args, adj=None, train_dates=("1871-01", "1972-12"), val_dates=("1973-01", "1983-12"),
         test_dates=("1984-01", "2020-08")):
    device = torch.device(args.device)
    args.device = device
    torch.set_num_threads(3)
    Data = DataLoaderS(args, train_dates=train_dates, val_dates=val_dates, test_dates=test_dates)
    args.num_nodes = Data.n_nodes
    args.save += f"_{Data.n_nodes}nodes.pt"
    print(Data, '\n')

    model = gtnet(args.gcn_true, args.adaptive_edges, args.gcn_depth, args.num_nodes, device, args,
                  predefined_A=adj, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels,
                  seq_length=args.window, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
    model = model.to(device)

    # print(args)
    print('The receptive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss().to(device)
    else:
        criterion = nn.MSELoss().to(device)
    evaluateL2 = nn.MSELoss().to(device)
    evaluateL1 = nn.L1Loss().to(device)

    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args)
            val_loss, val_rae, val_corr, oni_stats = \
                evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args)
            print(
                '--> Epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | Val. loss {:5.4f}, corr {:5.4f} |'
                ' ONI corr {:5.4f}, RMSE {:5.4f}'.format(epoch,
                                                         (time.time() - epoch_start_time),
                                                         train_loss, val_loss,
                                                         val_corr,
                                                         oni_stats["Corrcoef"],
                                                         oni_stats["RMSE"]), flush=True)
            # Save the model if the validation loss is the best we've seen so far.
            if oni_stats["RMSE"] < best_val:
                print("Model will be saved...")
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = oni_stats["RMSE"]
            if epoch % 5 == 0:
                test_acc, test_rae, test_corr, oni_stats = evaluate(Data, Data.test[0], Data.test[1], model,
                                                                    evaluateL2, evaluateL1, args)
                print("-------> Test stats: rse {:5.4f} | rae {:5.4f} | corr {:5.4f} |"
                      " ONI corr {:5.4f} | ONI RMSE {:5.4f}"
                      .format(test_acc, test_rae, test_corr, oni_stats["Corrcoef"], oni_stats["RMSE"]), flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f).to(device)

    val_acc, val_rae, val_corr, val_oni = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                   args)
    test_acc, test_rae, test_corr, oni_stats = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                        args)
    print("+++++++++++++++++++++  BEST MODEL STATS (best w.r.t to validations RMSE): +++++++++++++++++++++++++++++++")
    print("-------> Valid stats: rse {:5.4f} | rae {:5.4f} | corr {:5.4f} |"
          " ONI corr {:5.4f} | ONI RMSE {:5.4f}"
          .format(val_acc, val_rae, val_corr, val_oni["Corrcoef"], val_oni["RMSE"]), flush=True)
    print("-------> Test stats: rse {:5.4f} | rae {:5.4f} | corr {:5.4f} |"
          " ONI corr {:5.4f} | ONI RMSE {:5.4f}"
          .format(test_acc, test_rae, test_corr, oni_stats["Corrcoef"], oni_stats["RMSE"]), flush=True)
    print("Saved in", args.save)
    return val_acc, val_rae, val_corr, test_acc, test_rae, test_corr
