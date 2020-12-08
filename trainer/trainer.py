import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.mixup import mixup_data, mixup_criterion

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.train_metric_ftns = self.metric_ftns[0]
        self.val_metric_ftns = self.metric_ftns[1]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.val_imgs = self.valid_data_loader.dataset.imgs

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.train_metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.val_metric_ftns], writer=self.writer)
        
        #training-related config
        cfg_enhance = self.config['trainer_enhance']
        
        self.mixup = cfg_enhance['mixup']
        if self.mixup == True:
            self.mixup_alpha = cfg_enhance['mixup_alpha']        
                
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            
            # Mixup
            if self.mixup == True:
                inputs, targets_a, targets_b, lam = mixup_data(data, target,  
                                    alpha= self.mixup_alpha, use_cuda=torch.cuda.is_available())                
                # Forward pass
                output = self.model(inputs)

                # Loss
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(self.criterion, output)
            
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.train_metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
            
        #log = self.train_metrics.result()
        log = self.valid_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            #log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update(**{k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        all_output = []
        all_target = []
        #z_output = torch.zeros((512,8),dtype=torch.float32).to(self.device)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                #output = z_output
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                
                output_p = torch.nn.functional.log_softmax(output.detach(),1)            
                #output_p = z_output.detach()
                all_output.append(output_p)
                all_target.append(target.detach())
                #for met in self.val_metric_ftns:
                #    self.valid_metrics.update(met.__name__, met(output, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                
                #print(f'val batch: {batch_idx}')                
        # calculate accuracy on utterance level by averaging the posterior probabilities
        all_output = torch.cat(all_output,0)
        all_output = all_output.cpu().numpy()
        all_target = torch.cat(all_target,0)
        all_target = all_target.cpu().numpy()
        all_output = (all_output, all_target)
        
        for met in self.val_metric_ftns:
            self.valid_metrics.update(met.__name__, met(all_output, self.val_imgs))
        
        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
