import numpy as np
from torch.utils.data import DataLoader
from dataloader.dataset_seg import DataSet_RSseg, DataSet_Cityscapes


def CreateDataLoader(args):
    if 'ISPRS' in args.data_dir or 'LoveDA' in args.data_dir or 'DeepGlobe' in args.data_dir:
        train_lbl_dataset = DataSet_RSseg(args.data_dir, args.train_lbl_list, mode='train_l', ignore_index=args.ignore_index)    
        train_unl_dataset = DataSet_RSseg(args.data_dir, args.train_unl_list, mode='train_u', ignore_index=args.ignore_index)    
        val_dataset       = DataSet_RSseg(args.data_dir, args.val_list, mode='val', ignore_index=args.ignore_index)  

    train_lbl_loader = DataLoader(train_lbl_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=True, 
                                  drop_last=False,
                                  num_workers=args.num_workers, 
                                  pin_memory=False)    
    train_unl_loader = DataLoader(train_unl_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=True, 
                                  drop_last=False,
                                  num_workers=args.num_workers, 
                                  pin_memory=False)    
    val_loader       = DataLoader(val_dataset, 
                                  batch_size=12,
                                  shuffle=True, 
                                  drop_last=False,
                                  num_workers=args.num_workers, 
                                  pin_memory=False)    

    return train_lbl_loader, train_unl_loader, val_loader


def CreateTestDataLoader(dir_test, list_test, mode='val', batch_size=12):
    test_dataset = DataSet_RSseg(dir_test, list_test, mode=mode)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  drop_last=False,
                                  pin_memory=False)

    return test_dataloader


