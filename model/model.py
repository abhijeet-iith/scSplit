import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_count(model):
    count = count_parameters(model)
    print('')
    print('--------------------------------------------------')
    print(f'Number of trainable parameters: {count/1e6:.2f}M')
    print('--------------------------------------------------')
    print('')

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.best_val_metric = None
        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            if opt['train']['optimizer']['type'] == 'adam':
                self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']['optimizer']['type'] == 'adamax':
                self.optG = torch.optim.Adamax(optim_params, lr=opt['train']["optimizer"]["lr"], weight_decay=0)
            
            # set scheduler on PSNR, and therefore, it is max.
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optG,
                                                         'max',
                                                         patience=opt['train']['lr_scheduler']['patience'],
                                                         factor=0.2,
                                                         min_lr=1e-5,
                                                         verbose=True)
            print('Scheduler set to ReduceLROnPlateau with patience: ', opt['train']['lr_scheduler']['patience'])
            self.log_dict = OrderedDict()
        self.load_network()
        print_parameter_count(self.netG)
        # self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        # b, c, h, w = self.data['target'].shape
        # l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        diffusion_log = self.netG.get_current_log()
        for k, v in diffusion_log.items():
            self.log_dict[k] = v

    def test(self, continuous=False, clip_denoised=True):
        """
        It always tests on the synthetic input, never on the real input.
        """
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.prediction = self.netG.module.inference(
                    self.data['input'], 
                    # clip_denoised=clip_denoised,
                    continuous=continuous)
            else:
                if len(self.data['input'].shape) ==1:
                    ch1 = self.data['target'][:,:1]
                    ch2 = self.data['target'][:,1:]
                    inp1, inp2 = self.netG.get_xt_clean(
                        ch1,
                        ch2,
                        torch.Tensor([0.5]*len(ch1)).to(ch1.device),
                        update=False,
                    )
                else:
                    if self.netG.partial_step_training:
                        ch1 = self.data['target'][:, :1]
                        ch2 = self.data['target'][:, 1:2]
                        inp1, inp2 = self.netG.get_xt_clean(
                            ch1,
                            ch2,
                            torch.full(
                                (len(ch1),),
                                0.5,
                                device=ch1.device,
                            ),
                            update=False,
                        )
                    else:
                        ch1 = self.data['target']
                        ch2 = self.data['input']
                        inp1 = self.netG.get_xt_clean(
                            ch1,
                            ch2,
                            torch.Tensor([0.999]*len(ch1)).to(ch1.device),
                            update=False,
                        )
                # the hope is that with time, once the statistics are learned, both inp1 and inp2 will be the same.
                # normalization happens here. 
                if self.netG.partial_step_training:
                    self.prediction = self.netG.inference(
                        (inp1, inp2),
                        continuous=continuous,
                        t_float_start=0.5,
                    )
                else:
                    self.prediction = self.netG.inference(inp1,continuous=continuous)

        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.prediction = self.netG.module.sample(batch_size, continous)
            else:
                self.prediction = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.prediction.detach().float().cpu()
        else:
            out_dict['prediction'] = self.prediction.detach().float().cpu()
            out_dict['input'] = self.data['input'].detach().float().cpu()
            out_dict['target'] = self.data['target'].detach().float().cpu()
            # if need_LR and 'LR' in self.data:
            #     out_dict['LR'] = self.data['LR'].detach().float().cpu()
            # else:
            #     out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_best_network(self, epoch, iter_step, metric_val):
        if self.best_val_metric is None or metric_val > self.best_val_metric:
            self.best_val_metric = metric_val
        else:
            return False

        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
  
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        # delete the previous best model
        for f in os.listdir(self.opt['path']['checkpoint']):
            if f.startswith('best_gen'):
                os.remove(os.path.join(self.opt['path']['checkpoint'], f))
                break
            
        best_path = os.path.join(self.opt['path']['checkpoint'], 'best_gen_I{}_E{}_gen.pth.pth'.format(iter_step, epoch))
        torch.save(state_dict, best_path)
        logger.info('Saved best model in [{:s}] ...'.format(best_path))
      

    def save_network(self, epoch, iter_step):
        
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
  
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            # network.load_state_dict(torch.load(
            #     gen_path), strict=(not self.opt['model']['finetune_norm']))
            network.load_state_dict(torch.load(
                gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
