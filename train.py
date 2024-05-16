import os
import torch
import torch.nn.functional as F
from options.train_options import TrainOptions
from utils.util import Timer, AverageMeter, AverageVector
from dataloader import CreateDataLoader
from itertools import cycle
from models import CreateModel
import numpy as np
from evaluate import evaluate
import random
from utils.util import *
import cv2

torch.set_num_threads(4)
torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)


def main():
    _t = {'epoch_time': Timer()}
    _t['epoch_time'].tic()

    # get args parameters
    opt = TrainOptions()
    args = opt.initialize()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    percent = str(args.percent)
    
    args.train_lbl_list = args.train_lbl_list.replace('percent', percent)
    args.train_unl_list = args.train_unl_list.replace('percent', percent)
    args.data_dir = os.path.join(args.data_dir, args.dataset.replace('ISPRS_', 'ISPRS/'))
    method = args.method
    dataset = args.dataset
    opt.print_options(args)

    # create model
    torch.cuda.manual_seed(args.seed)
    model = CreateModel(args.model, args.num_classes).cuda()
    optimizer = torch.optim.Adam(model.optim_parameters(args), lr=args.learning_rate)

    # resume checkpoint
    args.save_dir = os.path.join(args.save_dir, args.model)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, args.method)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_file = '{}_{}_{}%.pth'.format(dataset, percent, str(args.alpha))
    checkpoint_file = os.path.join(save_dir, checkpoint_file)
    start_epoch = 0
    if os.path.exists(checkpoint_file) and args.restore:
        resume = torch.load(checkpoint_file)
        model.load_state_dict(resume['state_dict'])
        start_epoch = resume['epoch']
        best_acc = resume['best_acc']
        best_mIoU = resume['best_mIoU']
        print ('loading checkpoint from: {}, epoch: {}, best_mIoU: {} '\
                .format(checkpoint_file, start_epoch, best_mIoU))
    else:
        best_acc = 0
        best_mIoU = 0
        start_epoch = 0

    # build loader
    print ('Loading dataset ...')
    lbl_loader, unl_loader, val_loader = CreateDataLoader(args)
    iters, total_iters = 0, len(unl_loader)*args.num_epochs

    # initialize class-wise weight
    cls_memory_l  = {i: [torch.ones(1,args.num_classes).cuda()] for i in range(args.num_classes)}
    cls_memory_u  = {i: [torch.ones(1,args.num_classes).cuda()] for i in range(args.num_classes)}
    cls_memory_gt = {i: [torch.ones(1,args.num_classes).cuda()] for i in range(args.num_classes)}
    losses = AverageMeter()
    losses_l = AverageMeter()
    losses_u = AverageMeter()
    probs_l = AverageMeter()
    probs_u = AverageMeter()

    #-------------------------------------------------------------------#
    # Training, K-means, and Evaluation
    print ('Starting training ...')
    for epoch in range(start_epoch, args.num_epochs):
        model.adjust_learning_rate(args, optimizer, epoch, args.num_epochs)

        loader = zip(cycle(lbl_loader), unl_loader)
        for batch_idx, ((img_l, gt_l, [imgs_l_path, gts_l_path]), \
            (img_uw, img_us,  gt_u, [imgs_u_path, gts_u_path])) in enumerate(loader):
            # load data to GPU
            img_l, gt_l = img_l.cuda(), gt_l.cuda() 
            img_uw, img_us = img_uw.cuda(), img_us.cuda()
            b, _, h, w = img_l.size()

            # adjust learning rate
            iters = epoch*len(unl_loader) + batch_idx

            # batch_idx training
            optimizer.zero_grad()                                                 
            model.train()

            # sup loss
            logit_l = model(img_l)
            prob_l = torch.softmax(logit_l.detach(), dim=1)                
            loss_l = F.cross_entropy(logit_l, gt_l, ignore_index=args.ignore_index)
            loss = loss_l

            probs_l.update(prob_l.mean(dim=0).detach().cpu().numpy())  
            losses_l.update(loss_l.item(), b)  

            # unsuperivised loss
            if method == 'FixMatch':
                with torch.no_grad():
                    logit_uw = model(img_uw)   
                logit_us = model(img_us)   

                # fixmatch loss
                thr = args.confidence_thr
                prob_uw = torch.softmax(logit_uw.detach(), dim=1)
                max_prob, pseudo_gt = torch.max(prob_uw, dim=1)
                mask = max_prob.ge(thr).float()
                loss_u = (F.cross_entropy(logit_us, pseudo_gt, \
                            ignore_index=args.ignore_index, reduction='none')*mask).mean()
                
                loss = loss_l + loss_u  
                losses_u.update(loss_u.item(), b)  
                probs_u.update(prob_uw.mean(dim=0).detach().cpu().numpy())  

            if method == 'ST':
                with torch.no_grad():
                    logit_uw = model(img_uw)   
                logit_us = model(img_us)   

                # fixmatch loss
                thr = args.confidence_thr
                prob_uw = torch.softmax(logit_uw.detach(), dim=1)
                max_prob, pseudo_gt = torch.max(prob_uw, dim=1)
                loss_u = (F.cross_entropy(logit_us, pseudo_gt, \
                            ignore_index=args.ignore_index, reduction='none')).mean()
                
                loss = loss_l + loss_u  
                losses_u.update(loss_u.item(), b)  
                probs_u.update(prob_uw.mean(dim=0).detach().cpu().numpy())  

            if method == 'DWL':
                # get originial and downsampled logits
                with torch.no_grad():
                    logit_uw = model(img_uw)
                    logit_uw = logit_uw.detach()
                _, logit_us = model(img_us, return_pseudo_pred=True)

                # fixmatch loss
                prob_uw = torch.softmax(logit_uw.detach(), dim=1)
                max_prob_uw, pl_uw = torch.max(prob_uw, dim=1)  # N,  N
                probs_u.update(prob_uw.mean(dim=0).detach().cpu().numpy())  

                # distribution loss
                memory_n_batches = 50
                prob_l_bar = F.interpolate(prob_l, size=(64, 64), mode='nearest').permute(0,2,3,1).reshape(-1, args.num_classes)          
                prob_uw_bar = F.interpolate(prob_uw, size=(64, 64), mode='nearest').permute(0,2,3,1).reshape(-1, args.num_classes)          
                gt_l_bar = F.interpolate(gt_l.float().unsqueeze(dim=1), size=(64, 64), mode='nearest').squeeze(dim=1).reshape(-1).long()           
                pl_uw_bar = F.interpolate(pl_uw.float().unsqueeze(dim=1), size=(64, 64), mode='nearest').squeeze(dim=1).reshape(-1).long()           
                gt_u_bar = F.interpolate(gt_u.float().unsqueeze(dim=1), size=(64, 64), mode='nearest').squeeze(dim=1).reshape(-1).long()          
                cls_memory_l = update_cls_memory(cls_memory_l, prob_l_bar.detach(), gt_l_bar, memory_n_batches)
                cls_memory_u = update_cls_memory(cls_memory_u, prob_uw_bar.detach(), pl_uw_bar, memory_n_batches)
                cls_memory_gt = update_cls_memory(cls_memory_gt, prob_uw_bar.detach(), gt_u_bar, memory_n_batches)
                cls_bins_u  = sample_cls_bins(cls_memory_u)
                cls_bins_gt = sample_cls_bins(cls_memory_gt)

                logit_us = logit_us.permute(0,2,3,1).reshape(-1, args.num_classes)
                pl_uw, max_prob_uw = pl_uw.reshape(-1), max_prob_uw.reshape(-1)
                wgt_u = calc_wgt_bins(cls_bins_u, max_prob_uw, pl_uw, iters, total_iters)
                loss_u = (F.cross_entropy(logit_us, pl_uw, ignore_index=args.ignore_index, \
                                reduction='none')*wgt_u).mean() 
                if iters > memory_n_batches:
                    loss_u *= 1
                else:
                    loss_u *= 0
                loss = loss_l + loss_u  
                losses_u.update(loss_u.item(), b)  
                
            loss.backward()
            losses.update(loss.item(), b)
            optimizer.step()

            # print info
            if (batch_idx+1) % args.print_freq == 0:
                _t['epoch_time'].toc(average=False)
                if method=='Sup':
                    print ('[Epoch-Iter: [%d]: %d-%d][loss_l: %.4f][lr: %.4f][%.2fs]' % \
                            (epoch, batch_idx+1, total_iters, losses_l.avg,  
                                optimizer.param_groups[0]['lr']*1e4, _t['epoch_time'].diff) )
                else:
                    print ('[Epoch-Iter: [%d]: %d-%d][loss_l: %.4f][loss_u: %.4f][lr: %.4f][%.2fs]' % \
                            (epoch, batch_idx+1, total_iters, losses_l.avg, losses_u.avg,
                                optimizer.param_groups[0]['lr']*1e4, _t['epoch_time'].diff) )

        # evaluation and save
        val_acc, val_IoU = evaluate(args.num_classes, val_loader, model)
        val_mIoU = np.nanmean(np.delete(val_IoU, args.ignore_index, axis=0))
        print ('Val--  mIoU: {}, IoU: {}'.format(val_mIoU, val_IoU))
        if method == 'DistMatch' or method=='ST+WL':
            print ('Class Bins Pred: \n', cls_bins_u.detach().cpu().numpy())
            print ('Class Bins GT  : \n', cls_bins_gt.detach().cpu().numpy())

            # print ('Preds labeled:   ', probs_l.avg)
            # print ('Preds unlabeled: ', probs_u.avg)

        # reset the weight
        losses.reset()
        losses_l.reset()
        losses_u.reset()
        probs_l.reset()
        probs_u.reset()

        # save checkpoint
        if best_mIoU < val_mIoU:
            best_acc = val_acc
            best_mIoU = val_mIoU
            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'best_mIoU': best_mIoU,
                'state_dict': model.state_dict(),
            }
            torch.save(state, checkpoint_file)
            print ('taking snapshot ...')
            print ()

        record_dir = './records'
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)
        record_val_file = os.path.join(record_dir, args.method + '_' + args.dataset + '_' + str(args.percent) + '.txt')
        with open(record_val_file, 'a') as f:
            line = str(epoch) + ': '
            for element in val_IoU:
                line = line + str(element) + ' '
            line = line + str(val_acc)
            f.write(line + '\n')
        if method == 'ST+WL' or method == 'DistMatch':
            record_bin_file = os.path.join(record_dir, args.method + '_' + args.dataset + '_' + str(args.percent) + '_bin.txt')
            with open(record_bin_file, 'a') as f:
                np.savetxt(f, cls_bins_u.detach().cpu().numpy(), fmt='%0.5f') 
                f.write("---" + "\n")

        if method == 'ST+DL' or method == 'DWL':
            if args.model == 'SegFormer':
                for param, pseudo_param in zip(model.decoder.linear_pred.parameters(), model.decoder.pseudo_linear_pred.parameters()):
                    param.data = pseudo_param.data * args.alpha + param.data * (1-args.alpha)
            if args.model == 'HRNet':
                for param, pseudo_param in zip(model.last_layer.parameters(), model.pseudo_last_layer.parameters()):
                    param.data = pseudo_param.data * 0.5 + param.data * 0.5
            if args.model == 'DeepLab_V3plus':
                for param, pseudo_param in zip(model.classifier.parameters(), model.pseudo_classifier.parameters()):
                    param.data = pseudo_param.data * 0.5 + param.data * 0.5
            if args.model == 'EfficientUNet':
                for param, pseudo_param in zip(model.final_conv.parameters(), model.pseudo_final_conv.parameters()):
                    param.data = pseudo_param.data * 0.5 + param.data * 0.5

        _t['epoch_time'].tic()

if __name__ == '__main__':
    main()

