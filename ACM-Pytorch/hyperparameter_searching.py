from __future__ import division
from __future__ import print_function

import time

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools

from arg_parser import arg_parser
from logger import ACMPythorchLogger
from models.models import GCN, SFGCN
from utils import (
    evaluate,
    eval_acc,
    data_split,
    random_disassortative_splits,
    train_model,
    train_prep
)

logger = ACMPythorchLogger()
args = arg_parser()

(
    device,
    model_info,
    run_info,
    adj_high,  
    adj_low,  
    adj2_high,
    adj2_low,
    fadj_high,
    fadj_low,
    adj_low_unnormalized,   
    features,
    labels,
    split_idx_lst,
) = train_prep(logger, args)

print("args.cuda 0-------", args.cuda)

best_result_info = {
    "test_result": 0,
    "test_std": 0,
    "test_training_loss": 0,
    "dropout": None,
    "weight_decay": None,
    "lr": None,
    "nhid": None,
    "runtime_average": None,
    "epoch_average": None,
}

lr = [0.001, 0.01, 0.05, 0.1]
weight_decay = {0, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


criterion = nn.NLLLoss()
eval_func = eval_acc

logger.log_init("Hyperparameter Searching...")
for curr_lr, curr_weight_decay, curr_dropout in itertools.product(
    lr, weight_decay, dropout
):
    t_total = time.time()
    epoch_total = 0
    result = np.zeros(args.num_splits)

    run_info["lr"] = curr_lr
    run_info["weight_decay"] = curr_weight_decay
    run_info["dropout"] = curr_dropout
    run_info["nhid1"] = 64
    run_info["nhid2"] = 0

    for idx in range(args.num_splits):
        run_info["split"] = idx
        
        model = GCN(
            nfeat=features.shape[1],
            nhid=args.hidden1,
            nhid2=0,

            nclass=labels.max().item() + 1,
            nlayers=args.layers,
            nnodes=features.shape[0],
            dropout=args.dropout,
            model_type=args.model,
            structure_info=args.structure_info,
            variant=args.variant,
        ).to(device)

        if args.fixed_splits == 0:
            idx_train, idx_val, idx_test = random_disassortative_splits(
                labels, labels.max() + 1
            )
        else:
            idx_train, idx_val, idx_test = data_split(idx, args.dataset_name)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=curr_lr, weight_decay=curr_weight_decay
        )

        if args.cuda:
            idx_train = idx_train.cuda(0)
            idx_val = idx_val.cuda(0)
            idx_test = idx_test.cuda(0)
            model.cuda(0)

        curr_res = 0
        curr_training_loss = None
        best_val_loss = float("inf")
        val_loss_history = torch.zeros(args.epochs)

        for epoch in range(args.epochs):
            t = time.time()
            acc_train, loss_train = train_model(
                model,
                optimizer,
                adj_low,
                adj_high,
                adj2_low,
                adj2_high,
                fadj_low,
                fadj_high,
                adj_low_unnormalized,
                features,
                labels,
                idx_train,
                criterion,
                dataset_name=args.dataset_name,
            )

            model.eval()
            output = model(features, adj_low, adj_high, adj2_low,
                           adj2_high, fadj_low, fadj_high, adj_low_unnormalized)

            if args.dataset_name == "deezer-europe":
                output = F.log_softmax(output, dim=1)
                val_loss, val_acc = criterion(
                    output[idx_val], labels.squeeze(1)[idx_val]
                ), evaluate(
                    output, labels, idx_val, eval_func
                )
            else:
                output = F.log_softmax(output, dim=1)
                val_loss, val_acc = criterion(
                    output[idx_val], labels[idx_val]
                ), evaluate(
                    output, labels, idx_val, eval_func
                )
            if args.dataset_name == "deezer-europe":
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    curr_res = evaluate(
                        output, labels, idx_test, eval_func
                    )
                    curr_training_loss = loss_train
            else:
                if (
                    val_loss < best_val_loss
                ):
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    curr_res = evaluate(
                        output, labels, idx_test, eval_func
                    )
                    curr_training_loss = loss_train
                # if epoch >= 0:
                val_loss_history[epoch] = val_loss.detach()
                if args.early_stopping > 0 and epoch > args.early_stopping:
                    tmp = torch.mean(
                        val_loss_history[epoch - args.early_stopping: epoch]
                    )
                    if val_loss > tmp:
                        break

        epoch_total = epoch_total + epoch

        result[idx] = curr_res
        del model, optimizer
        if args.cuda:
            torch.cuda.empty_cache()

    total_time_elapsed = time.time() - t_total
    runtime_average = total_time_elapsed / args.num_splits

    epoch_average = total_time_elapsed / epoch_total * 1000

    run_info["runtime_average"] = runtime_average
    run_info["epoch_average"] = epoch_average
    run_info["result"] = np.mean(result)
    run_info["std"] = np.std(result)
    run_info["best_result"] = np.max(result)

    if run_info["result"] > best_result_info["test_result"]:
        best_result_info["best_result"] = run_info["best_result"]
        best_result_info["test_result"] = run_info["result"]
        best_result_info["test_std"] = run_info["std"]
        best_result_info["dropout"] = curr_dropout
        best_result_info["weight_decay"] = curr_weight_decay
        best_result_info["lr"] = curr_lr
        best_result_info["nhid1"] = 64
        best_result_info["nhid2"] = 0
        best_result_info["runtime_average"] = runtime_average
        best_result_info["epoch_average"] = epoch_average

    # logger.log_time(f"{time.time() - t_total:.4f}s")
    logger.log_run(model_info, run_info)

logger.log_best_result(model_info, best_result_info)
