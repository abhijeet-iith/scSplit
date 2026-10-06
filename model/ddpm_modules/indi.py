import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import torch

from model.ddpm_modules.diffusion import GaussianDiffusion, exists, default, make_beta_schedule
from model.normalizer import NormalizerXT


class InDI(GaussianDiffusion):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        out_channel=2,
        lr_reduction=None,
        conditional=True,
        schedule_opt=None,
        val_schedule_opt=None,
        e = 0.01,
        time_predictor=None,
        xt_normalizer=None,
        partial_step_training=False,
    ):
        super().__init__(denoise_fn, image_size, channels=channels, loss_type=loss_type, conditional=conditional, 
                         lr_reduction=lr_reduction,
                         schedule_opt=schedule_opt,
                         )
        self.e = e
        self.out_channel = out_channel
        self._t_sampling_mode = 'linear_indi'
        assert self._t_sampling_mode in ['uniform', 'linear_ramp', 'quadratic_ramp', 'linear_indi']
        self._linear_indi_a = 1.0

        self._noise_mode = 'gaussian'
        assert self._noise_mode in ['gaussian', 'brownian', 'none']
        if self._noise_mode == 'none':
            self.e = 0.0
        
        self.val_num_timesteps = val_schedule_opt['n_timestep']
        self.time_predictor = time_predictor
        self._xt_normalizer = xt_normalizer
        self._normalize_xt = self._xt_normalizer is not None
        self.partial_step_training = partial_step_training

        msg = f'Sampling mode: {self._t_sampling_mode}, Noise mode: {self._noise_mode}'
        print(f'[{self.__class__.__name__}]: {msg}')

    def set_new_noise_schedule(self, schedule_opt, device):
        self.num_timesteps= schedule_opt['n_timestep']
    

    def q_mean_variance(self, x_start, t):
        raise NotImplementedError("This is not needed.")
    
    def predict_start_from_noise(self, x_t, t, noise):
        raise NotImplementedError("This is not needed.")
    
    def q_posterior(self, x_start, x_t, t):
        raise NotImplementedError("This is not needed.")
    
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        raise NotImplementedError("This is not needed.")

    @torch.no_grad()
    def inference_one_step(self, x_t, delta_t, t_cur):
        assert delta_t <= t_cur, f"delta_t:{delta_t} should be less than or equal to t_cur:{t_cur}."
        t_cur = torch.Tensor([t_cur]).to(x_t.device)
        x_0 = self.denoise_fn(x_t, t_cur)
        noise = torch.randn_like(x_t) * self.get_t_times_e(t_cur-delta_t)
        x_prev_t = delta_t/t_cur * x_0 + (1 - delta_t/t_cur) * x_t + noise
        return x_prev_t

    @torch.no_grad()
    def inference(self, x_in, continuous=False, num_timesteps=None, t_float_start=1.0, eps=1e-8):
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        device = x_in.device
        sample_inter = (1 | (num_timesteps//20))
        assert self.conditional is False
        # b = x_in.shape[0]
        # assert x_in.shape[1] == 1, "Only one channel is supported."

        # factor = self.out_channel // x_in.shape[1]
        # x_in = torch.cat([x_in]*factor, dim=1)
        x_t = x_in + torch.randn_like(x_in)*self.get_t_times_e(torch.Tensor([t_float_start]).to(device))
        delta = t_float_start / num_timesteps
        cur_t = t_float_start
        ret_img = x_t
        for idx in range(num_timesteps):
            x_t = self.inference_one_step(x_t, delta, cur_t)
            cur_t -= delta
            if idx % sample_inter == 0 or idx == num_timesteps-1:
                ret_img = torch.cat([ret_img, x_t], dim=0)
        
        if continuous:
            return ret_img
        else:
            return ret_img[-1:]
    

    def get_e(self, t):
        # TODO: for brownian motion, this will change.
        if self._noise_mode in ['gaussian', 'none']:
            return self.e
        elif self._noise_mode == 'brownian':
            assert t > 0, "t must be non-negative."
            return self.e/torch.sqrt(t)
        
    def get_t_times_e(self, t):
        if self._noise_mode in ['gaussian', 'none']:
            return self.e * t
        elif self._noise_mode == 'brownian':
            return self.e * torch.sqrt(t)
    
    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        raise NotImplementedError("This is not needed.")
    
    def normalize_xt(self, x_t, t, update=True):
        if self._xt_normalizer is None:
            return x_t
        
        return self._xt_normalizer.normalize(x_t, t, update=update)
    

    def get_xt_clean(self, x_start, x_end, t:float, update=True):
        """
        Returns an estimate of the superimposed input at time t.
        Essentially, this is the interpolation between x_start and x_end at time t.
        Optionally, it can normalize the output as well.
        """
        assert 0 <= t.min(), "t >= 0"
        assert t.max() <= 1, "t <= 1. but t is {}".format(t.max())

        if len(t.shape) ==1:
            t = t.reshape(-1, 1, 1, 1)

        x_t_clean = (1-t)*x_start + t*x_end
        # normalization
        if self._normalize_xt:
            x_t_clean = self._xt_normalizer.normalize(x_t_clean, t, update=update)
        return x_t_clean
    
        
    def sample_t(self, batch_size, device):
        if self._t_sampling_mode == 'linear_ramp':
            # probablity of t=0 is 0, which is what we want.
            probablity =torch.arange(self.num_timesteps)
            probablity = probablity/torch.sum(probablity)
            t = torch.multinomial(probablity,batch_size,replacement=True).to(device).long()
        elif self._t_sampling_mode == 'quadratic_ramp':
            # probablity of t=0 is 0, which is what we want.
            probablity =torch.arange(self.num_timesteps)**2
            probablity = probablity/torch.sum(probablity)
            t = torch.multinomial(probablity,batch_size,replacement=True).to(device).long()
        elif self._t_sampling_mode == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (batch_size,),device=device).long()
        elif self._t_sampling_mode == 'uniform_in_range':
            t = torch.randint((2*self.num_timesteps)//3, self.num_timesteps+1, (batch_size,),device=device).long()
        elif self._t_sampling_mode == 'linear_indi':
            maxv = self.num_timesteps
            t = torch.randint(1, maxv, (batch_size,),device=device).long()
            alpha = 1/(self._linear_indi_a + 1)
            probab = torch.rand(t.shape, device=device)
            mask_for_max = probab > alpha
            t[mask_for_max] = maxv
        
        t_float = t/self.num_timesteps
        return t_float

    def time_prediction_loss(self, x_clean, actual_t):
        predicted_t = self.time_predictor(x_clean)
        # print(predicted_t.detach().cpu().numpy())
        # print(actual_t.detach().cpu().numpy())
        # print('---')
        loss = F.mse_loss(predicted_t, actual_t)
        return loss, predicted_t
    
    def get_prediction_during_training(self, x_in, noise=None, use_superimposed_input=False):
        # we want to make sure that the shape for x_end is the same as x_start.
        target = None
        target_t = None
        if use_superimposed_input:
            x_superimposed_input = x_in['superimposed_input'] # this is already some mixture of x_start and x_end. it can come from microscope
            x_clean = x_superimposed_input
            t_float = self.time_predictor(x_clean)
            step_float = t_float
        else:
            x_start = x_in['target']
            x_end = x_in['input']
            # factor = self.out_channel // x_end.shape[1]
            # x_end = torch.concat([x_end]*factor, dim=1)
            b, *_ = x_start.shape
            t_float = self.sample_t(b, x_start.device)
            x_clean = self.get_xt_clean(x_start=x_start, x_end=x_end, t=t_float)
            if self.partial_step_training:
                step_float = torch.rand_like(t_float) * t_float
                target_t = t_float - step_float
                target = self.get_xt_clean(
                    x_start=x_start,
                    x_end=x_end,
                    t=target_t,
                )
                target = self.normalize_xt(target, target_t, update=False)
            else:
                step_float = t_float
                target_t = torch.zeros_like(t_float)
                target = None

        noise = default(noise, lambda: torch.randn_like(x_clean))
        x_noisy = x_clean + noise * self.get_t_times_e(t_float).reshape(-1, 1, 1, 1)

        assert self.conditional is False
        conditioning_time = step_float if self.partial_step_training else t_float
        x_recon = self.denoise_fn(x_noisy, conditioning_time)
        return x_recon, {
            't_float': t_float,
            'b_float': step_float,
            'target_t': target_t,
            'x_clean': x_clean,
            'target': target,
        }

    def p_losses(self, x_in, noise=None):
        x_start = x_in['target']
        x_recon, prediction_dict = self.get_prediction_during_training(x_in, noise=noise)
        device = x_start.device
        if self.partial_step_training:
            target = prediction_dict['target']
        else:
            target = self._xt_normalizer.normalize(x_start, torch.Tensor([0.0]*len(x_start)).to(device), update=False)
        loss = self.loss_func(target, x_recon)

        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
