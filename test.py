import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import os
from model import model_dict
from datasets import dataset_dict
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, dataloader
from torch.utils.data.distributed import DistributedSampler
from testengine import train_one_epoch, evaluate, quantize_per_tensor
from torch.utils.tensorboard import SummaryWriter
import utils.misc as utils
import torch.nn.utils.prune as prune
import torch.quantization

def get_args_parse():
    parser = argparse.ArgumentParser('Dense NeRV', add_help=False)

    parser.add_argument('--cfg_path', default='', type=str, help='path to specific cfg yaml file path')
    parser.add_argument('--output_dir', default='', type=str, help='path to save the log and other files')
    parser.add_argument('--time_str', default='', type=str, help='just for tensorboard dir name')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--port', default=29500, type=int, help='port number')
    parser.add_argument('--rank', default=None, type=int)

    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')

    # pruning paramaters
    parser.add_argument('--prune_ratio', type=float, default=1.0, help='pruning ratio')
    parser.add_argument('--prune_steps', type=float, nargs='+', default=[0.,], help='prune steps')

    parser.add_argument('--quant_bit', type=int, default=-1, help='bit length for model quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')

    parser.add_argument('--save_image', action='store_true', default=False, help='dump the prediction images')


    return parser


def main(args):
    utils.init_distributed_mode(args)
    print('git:\n {}\n'.format(utils.get_sha()))

    # get cfg yaml file
    cfg = utils.load_yaml_as_dict(args.cfg_path)
    # dump the cfg yaml file in output dir
    utils.dump_cfg_yaml(cfg, args.output_dir)
    print(cfg)

    device = torch.device(args.device)

    # fix the seed
    seed = cfg['seed']
    # seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = model_dict[cfg['model']['model_name']](cfg=cfg['model'])
    model.to(device)

    model_without_ddp = model

    loc = 'cuda:0'
    args.epochs = cfg['epoch']

    local_rank = None
    ##### prune model params and flops #####
    if args.prune_ratio < 1 and not args.eval_only: 
        prune_str = '_Prune{}_{}'.format(args.prune_ratio, ','.join([str(x) for x in args.prune_steps]))
    else:
        prune_str = ''
    print(prune_str)
    prune_net = args.prune_ratio < 1
    if prune_net:
        param_list = []
        for k,v in model.named_parameters():
            # print(k)
            if 'weight' in k:
                # print(k)
                if 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    if "xy" in k:
                        param_list.append(model.stem_xy[stem_ind])
                    else:
                        param_list.append(model.stem_t[stem_ind])
                elif 't_layers' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.t_layers[layer_ind][0])
                elif 'layers' in k[:6] and 'conv' in k:
                    layer_ind = int(k.split('.')[1])
                    if 'conv1' in k:
                        param_list.append(model.layers[layer_ind].conv1.conv)
                        # print(model.layers[layer_ind].conv1.conv)

                    elif 'conv2' in k:
                        param_list.append(model.layers[layer_ind].conv2)
                        # print(model.layers[layer_ind].conv2)

                    else:
                        # print(model.layers[layer_ind].conv.conv)
                        param_list.append(model.layers[layer_ind].conv.conv)
                
                elif 'trans' in k:
                    # trans_ind = k[5]
                    # print(k[5])
                    if int(k[5]) == 1:
                        # print("inin")
                        if 'to_qkv' in k :
                            # print(model.trans1.attn.to_qkv)

                            param_list.append(model.trans1.attn.to_qkv)
                        elif 'to_out' in k :
                            # continue
                            #trans1.attn.to_out.0.weight
                            # Sequential(
                            #     (0): Linear(in_features=64, out_features=256, bias=True)
                            #     (1): Dropout(p=0.0, inplace=False)
                            # )
                            # print(model.trans1.attn.to_out)
                            param_list.append(model.trans1.attn.to_out[0])
                        else:
                            trans_id = int(k.split('.')[3])
                            # print(model.trans1.ffn.net[trans_id])

                            param_list.append(model.trans1.ffn.net[trans_id])
                    else:
                        if 'to_qkv' in k :
                            param_list.append(model.trans2.attn.to_qkv)
                        elif 'to_out' in k :
                            # continue
                            param_list.append(model.trans2.attn.to_out[0])
                        else:
                            trans_id = int(k.split('.')[3])
                            param_list.append(model.trans2.ffn.net[trans_id])
                elif 'toconv' in k:
                    # print(model.toconv[0])
                    param_list.append(model.toconv[0])
                elif 'head_layers' in k:
                    # continue

                    param_list.append(model.head_layers[4])
                elif 't_branch' in k :

                    branch_id = int(k.split('.')[1])

                    param_list.append(model.t_branch[branch_id])

        print(param_list)

        param_to_prune = [(ele, 'weight') for ele in param_list]
        print(param_to_prune)

        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
        prune_num = 0
        if args.eval_only:
            print("eval only prune")
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
    ##### get model params and flops #####
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    print("total_params",total_params)
    # get model params
    if args.rank in [0, None]:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
        print(f'{args}\n {model}\n Model Params: {params}M')
        tmpparam = utils.get_n_params(model)
        tmpbpp = (tmpparam*32) / (133 * 1280 * 720)
        print(f'cur parameters : {tmpparam}')
        print(f'cur bpp : {tmpbpp}')
        # tensorboard writer
        writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard_{}'.format(args.time_str)))
    else:
        writer = None
    

    img_transform = transforms.ToTensor()
    print("dataset_train")

    dataset_train = dataset_dict[cfg['dataset_type']](main_dir=cfg['dataset_path'], transform=img_transform, train=True)
    print("dataset_val")

    dataset_val = dataset_dict[cfg['dataset_type']](main_dir=cfg['dataset_path'], transform=img_transform, train=False)
    # follow nerv implementation on sampler and dataloader
    sampler_train = DistributedSampler(dataset_train) if args.distributed else None
    sampler_val = DistributedSampler(dataset_val) if args.distributed else None
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=cfg['train_batchsize'], shuffle=(sampler_train is None), num_workers=cfg['workers'], 
        pin_memory=True, sampler=sampler_train, drop_last=True, worker_init_fn=utils.worker_init_fn
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=cfg['val_batchsize'], shuffle=False, num_workers=cfg['workers'], 
        pin_memory=True, sampler=sampler_val, drop_last=False, worker_init_fn=utils.worker_init_fn
    )

    datasize = len(dataset_train)
    
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad],
            "lr": cfg['optim']['lr'],
        }
    ]
    print("param_dicts")

   
    

    optim_cfg = cfg['optim']
    if optim_cfg['optim_type'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg['lr'], betas=(optim_cfg['beta1'], optim_cfg['beta2']))
    elif optim_cfg['optim_type'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),lr=optim_cfg['lr'], betas=(optim_cfg['beta1'], optim_cfg['beta2']), eps=1e-08)
    else:
        optimizer = None
    assert optimizer is not None, "No implementation of Optimizer!"

    if args.distributed:
        print("distributed")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    output_dir = Path(args.output_dir)

    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]

    # resume from args.weight
    checkpoint = None
    # loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path)
        orig_ckt = checkpoint['model']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt)
        else:
            model.load_state_dict(new_ckt, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        
    # resume from model_latest
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch Init: {sparisity_num / 1e6 / total_params}')
        model.load_state_dict(checkpoint['model'] ,  strict=False)
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))
    args.start_epoch = 0
    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch'] 
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
        print('load optimizer')    
        optimizer.load_state_dict(checkpoint['optimizer'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()
    if args.not_resume_epoch:
        args.start_epoch = 0
    print('Start training')
    start_time = datetime.now()




    if args.eval_only:
        print('Evaluation ...')
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        print_str = f'{time_str}\t Results for checkpoint: {args.weight}\n'
        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print_str += f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}\n'
        
            model_sparsity =  float(sparisity_num / 1e6 / total_params)   
            print("model_sparsity",model_sparsity)
            bpp = (total_params*8000000*(args.prune_ratio)*args.quant_bit/(133*720*1280))
            print(f'bpp : {bpp}\n')
            print_str += f'bpp : {bpp}\n'
        # import pdb; pdb.set_trace; from IPython import embed; embed()
        # val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)

        #quantization and entropy coding
        if args.quant_bit != -1:

            
            cur_ckt = model.state_dict()
            from dahuffman import HuffmanCodec
            quant_weitht_list = []
            for k,v in cur_ckt.items():
                if ('stem' in k or 'layers' in k):
                    large_tf = (v.dim() in {2,4} and 'bias' not in k)
                    quant_v, new_v = quantize_per_tensor(v, args.quant_bit, args.quant_axis if large_tf else -1)
                    valid_quant_v = quant_v[v!=0] # only include non-zero weights
                    quant_weitht_list.append(valid_quant_v.flatten())
                else:
                    new_v = v
                cur_ckt[k] = new_v
            cat_param = torch.cat(quant_weitht_list)
            input_code_list = cat_param.tolist()
            unique, counts = np.unique(input_code_list, return_counts=True)
            num_freq = dict(zip(unique, counts))

            # generating HuffmanCoding table
            codec = HuffmanCodec.from_data(input_code_list)
            sym_bit_dict = {}
            for k, v in codec.get_code_table().items():
                sym_bit_dict[k] = v[0]
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            avg_bits = total_bits / len(input_code_list)    
            # import pdb; pdb.set_trace; from IPython import embed; embed()       
            encoding_efficiency = avg_bits / args.quant_bit
            print_str = f'Entropy encoding efficiency for bit {args.quant_bit}: {encoding_efficiency}'
            print(print_str)
            # if local_rank in [0, None]:
            #     with open('{}/eval.txt'.format(args.outf), 'a') as f:
            #         f.write(print_str + '\n')       
            model.load_state_dict(cur_ckt)

            # import pdb; pdb.set_trace; from IPython import embed; embed()

        save_result  =  True if args.save_image == True else False
        val_stats = evaluate(
            model, dataloader_val, device, cfg, args, save_image=save_result  # TODO: implement the save image
        )

        if prune_net:
            val_best_psnr = val_stats['val_psnr'][-1]
            val_best_msssim = val_stats['val_msssim'][-1]
        else:
        
            val_best_psnr = val_stats['val_psnr'][-1] if val_stats['val_psnr'][-1] > val_best_psnr else val_best_psnr
            val_best_msssim = val_stats['val_msssim'][-1] if val_stats['val_msssim'][-1] > val_best_msssim else val_best_msssim
        # if args.rank in [0, None]:
        #     print_str = f'Eval best_PSNR at epoch{epoch+1}:'
        #     print_str += '\tevaluation: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}'.format(
        #         val_stats['val_psnr'][-1].item(), val_best_psnr.item(), val_best_msssim.item())
        #     print(print_str)
        print_str += f'PSNR/ms_ssim on validate set for bit {args.quant_bit} with axis {args.quant_axis}: {round(val_best_psnr.item(),2)}/{round(val_best_msssim.item(),4)}'
        print(print_str)
        # with open('{}/eval.txt'.format(args.outf), 'a') as f:
        #     f.write(print_str + '\n\n')        
        return

    
    for epoch in range(args.start_epoch,cfg['epoch']):
        print("epoch"+str(epoch))
        if args.distributed:
            sampler_train.set_epoch(epoch)
        model.train()
        print(prune_net ,args.prune_steps)
        if prune_net and epoch in args.prune_steps:
            print("inprune")
            prune_num += 1 
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{epoch}: {sparisity_num / 1e6 / total_params}')
        train_stats = train_one_epoch(
            model, dataloader_train, optimizer, device, epoch, cfg, args, datasize, start_time, writer 
        )
        
        train_best_psnr = train_stats['train_psnr'][-1] if train_stats['train_psnr'][-1] > train_best_psnr else train_best_psnr
        train_best_msssim = train_stats['train_msssim'][-1] if train_stats['train_msssim'][-1] > train_best_msssim else train_best_msssim
        if args.rank in [0, None]:
            print_str = '\ttraining: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(train_stats['train_psnr'][-1].item(), 
            train_best_psnr.item(), train_best_msssim.item())
            print(print_str, flush=True)
        checkpoint_paths = [output_dir / 'checkpoint.pth']  # save one per epoch
        print("save" , checkpoint_paths)

        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': cfg,
                'train_best_psnr': train_best_psnr,
                'train_best_msssim': train_best_msssim,
                'val_best_psnr': val_best_psnr,
                'val_best_msssim': val_best_msssim,
            }, checkpoint_path)
        
        # evaluation
        if (epoch + 1) % cfg['eval_freq'] == 0 or epoch > cfg['epoch'] - 10:
            val_stats = evaluate(
                model, dataloader_val, device, cfg, args, save_image=False  # TODO: implement the save image
            )
            
            val_best_psnr = val_stats['val_psnr'][-1] if val_stats['val_psnr'][-1] > val_best_psnr else val_best_psnr
            val_best_msssim = val_stats['val_msssim'][-1] if val_stats['val_msssim'][-1] > val_best_msssim else val_best_msssim
            if args.rank in [0, None]:
                print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                print_str += '\tevaluation: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}'.format(
                    val_stats['val_psnr'][-1].item(), val_best_psnr.item(), val_best_msssim.item())
                print(print_str)
    print("Training complete in: " + str(datetime.now() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('E-NeRV training and evaluation script', parents=[get_args_parse()])
    args = parser.parse_args()

    assert args.cfg_path is not None, 'Need a specific cfg path!'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)