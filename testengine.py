import math
import os
import sys
import torch
import utils.misc as utils
import torch.nn.functional as F
from datetime import datetime
import torch.nn.utils.prune as prune

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    cfg,
    args,
    datasize,
    start_time,

    writer=None,
    
):
    
    torch.cuda.empty_cache()

    epoch_start_time = datetime.now()
    loss_type = cfg["loss"]

    psnr_list = []
    msssim_list = []
    all_loss_list = []
    # print(list(torch.utils.data.DataLoader()))
    for i, data in enumerate(dataloader):
        data = utils.to_cuda(data, device)
        # print(data['img_gt'].shape)
        # forward pass
        # print(f' {model}\n ')
        output_list = model(data)  # output is a list for the case that has multiscale
        
        additional_loss_item = {}
        if isinstance(output_list, dict):
            for k, v in output_list.items():
                if "loss" in k:
                    additional_loss_item[k] = v
            output_list = output_list["output_list"]
        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list
        ]
        loss_list = utils.loss_compute(output_list, target_list, loss_type)
        losses = sum(loss_list)
        if len(additional_loss_item.values()) > 0:
            losses = losses + sum(additional_loss_item.values())
        all_loss_list.append(losses)
        # print(losses)
        lr = utils.adjust_lr(optimizer, epoch, cfg["epoch"], i, datasize, cfg)
        optimizer.zero_grad()
        losses.backward()
        # print(optimizer)
        optimizer.step()

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))
        msssim_list.append(utils.msssim_fn(output_list, target_list))

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:
            train_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            train_psnr = torch.mean(train_psnr, dim=0)  # (num_stage)
            train_msssim = torch.cat(msssim_list, dim=0)  # (batchsize, num_stage)
            train_msssim = torch.mean(train_msssim.float(), dim=0)  # (num_stage)
            time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            if not hasattr(args, "rank"):
                print_str = "[{}] Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    epoch + 1,
                    cfg["epoch"],
                    i + 1,
                    len(dataloader),
                    lr,
                    utils.RoundTensor(train_psnr, 2, False),
                    utils.RoundTensor(train_msssim, 4, False),
                )
                for k, v in additional_loss_item.items():
                    print_str += f", {k}: {v.item():.6g}"
                print(print_str, flush=True)

            elif args.rank in [0, None]:
                print_str = "[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    args.rank,
                    epoch + 1,
                    cfg["epoch"],
                    i + 1,
                    len(dataloader),
                    lr,
                    utils.RoundTensor(train_psnr, 2, False),
                    utils.RoundTensor(train_msssim, 4, False),
                )
                print(print_str, flush=True)

    train_stats = {
        "train_psnr": train_psnr,
        "train_msssim": train_msssim,
    }
    if hasattr(args, "distributed") and args.distributed:
        train_stats = utils.reduce_dict(train_stats)

    # ADD train_PSNR TO TENSORBOARD
    if not hasattr(args, "rank"):
        h, w = output_list[-1].shape[-2:]
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
        writer.add_scalar("Train/loss", losses, epoch + 1)
        
        for k, v in additional_loss_item.items():
            writer.add_scalar(f"Train/{k}", v.item(), epoch + 1)
        for (k, m) in model.named_modules():
            if isinstance(m, torch.nn.Module) and hasattr(m, "Lip_c"):
                writer.add_scalar(f"Stat/{k}_c", m.Lip_c[0].item(), epoch + 1)
                writer.add_scalar(f"Stat/{k}_w", m.abssum_max, epoch + 1)

    elif args.rank in [0, None] and writer is not None:
        h, w = output_list[-1].shape[-2:]
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
        writer.add_scalar("Train/loss", all_loss_list[-1].item(), epoch + 1)
    epoch_end_time = datetime.now()
    print(
        "Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format(
            (epoch_end_time - epoch_start_time).total_seconds(),
            (epoch_end_time - start_time).total_seconds() / (epoch + 1),
        )
    )

    return train_stats


@torch.no_grad()
def evaluate(model, dataloader, device, cfg, args, save_image=False):

    if save_image:
        from torchvision.utils import save_image
        visual_dir = f'{args.output_dir}/visualize'
        print(f'Saving predictions to {visual_dir}')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)

    
    val_start_time = datetime.now()
    model.eval()

    psnr_list = []
    msssim_list = []

    for i, data in enumerate(dataloader):
        data = utils.to_cuda(data, device)
        # print(data)
        # forward pass
        output_list = model(data)  # output is a list for the case that has multiscale
        if isinstance(output_list, dict):
            output_list = output_list["output_list"]  # ignore the loss in eval
        torch.cuda.synchronize()
        if save_image:
            for batch_ind in range(1):
                full_ind = i * 1 + batch_ind
                save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                save_image(data["img_gt"], f'{visual_dir}/gt_{full_ind}.png')

        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list
        ]

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))
        msssim_list.append(utils.msssim_fn(output_list, target_list))

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:
            val_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            val_psnr = torch.mean(val_psnr, dim=0)  # (num_stage)
            val_msssim = torch.cat(msssim_list, dim=0)  # (batchsize, num_stage)
            val_msssim = torch.mean(val_msssim.float(), dim=0)  # (num_stage)
            time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            if not hasattr(args, "rank"):
                print_str = "[{}], Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                print(print_str, flush=True)

            elif args.rank in [0, None]:
                print_str = "[{}] Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    time_now_string,
                    args.rank,
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                print(print_str, flush=True)

    val_stats = {
        "val_psnr": val_psnr,
        "val_msssim": val_msssim,
    }
    if hasattr(args, "distributed") and args.distributed:
        val_stats = utils.reduce_dict(val_stats)
    val_end_time = datetime.now()
    print(
        "Time on evaluate: \t{:.2f}".format(
            (val_end_time - val_start_time).total_seconds()
        )
    )

    return val_stats


def quantize_per_tensor(t, bit=8, axis=-1):
    if axis == -1:
        t_valid = t!=0
        t_min, t_max =  t[t_valid].min(), t[t_valid].max()
        scale = (t_max - t_min) / 2**bit
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i]!=0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)        
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[:,None,None,None]
            t_min = min_max_tf[:,0,None,None,None]
        elif t.dim() == 2:
            scale = scale[:,None]
            t_min = min_max_tf[:,0,None]
    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:,i]!=0
            if t_valid.sum():
                min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)             
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None,:,None,None]
            t_min = min_max_tf[None,:,0,None,None]
        elif t.dim() == 2:
            scale = scale[None,:]
            t_min = min_max_tf[None,:,0]            
    # import pdb; pdb.set_trace; from IPython import embed; embed()       
    quant_t = ((t - t_min) / (scale + 1e-19)).round()
    new_t = t_min + scale * quant_t
    return quant_t, new_t