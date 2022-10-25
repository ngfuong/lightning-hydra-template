import datetime
import os
import logging

import torch
from tensorboardX import SummaryWriter


class Logger:
    r"""Writes evaluation resultsw of training/testing"""
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '':
            logpath = 'log' + logtime
        
        cls.logpath = os.path.join('logs', logpath, 'log')
        if not os.path.exists(cls.logpath):
            os.makedirs(cls.logpath)
        
        logging.basicConfig(
            filemode='w',
            filename=os.path.join(cls.logpath, 'log.txt'), 
            level=logging.INFO,
            format='%(message)%s',
            datefmt='%m-%d %H:%M:%S'
            )
        
        # Console lo config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)')
        console.setFormatter(formatter)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tensorboard/runs'))

        # Log arguments
        logging.info("\n:========== Fashion Visual Search ==========")
        for arg in args.__dict__:
            logging.info('| %20: %-24s' % (arg, str(args.__dict__[arg])))
        logging.info("\n:===========================================")

    @classmethod
    def info(cls, msg):
        r"""Writes message to log.txt"""
        logging.info(msg)
    
    @classmethod
    def save_model_state(cls, model, epoch, val_loss):
        r"""Saves model state to logpath"""
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best.pth'))
        cls.info('Model epoch = %03d with Val Loss: %5.2f saved to %s' %(epoch, val_loss, os.path.join(cls.logpath, 'best.pth')))
    
    @classmethod
    def log_params(cls, model):
        backbone_param = 0 
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] == 'backbone':
                if k.split('.')[1] in ['fc', 'classifier']:
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        
        Logger.info('Backbone #param.: %d' %backbone_param)
        Logger.info('Learnable #param.: %d' %learner_param)
        Logger.info('Total #param.: %d' %(backbone_param + learner_param))
    
    