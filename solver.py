import os
import time
import shutil
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate
from stft import mag_pha_istft, mag_pha_stft
from utils import copy_state, swap_state,  \
                  batch_pesq, phase_losses, LogProgress, pull_metric
            

class Solver(object):
    def __init__(
        self, 
        data, 
        model,
        discriminator,
        optim, 
        optim_disc,
        scheduler,
        scheduler_disc,
        args, 
        logger, 
        device=None
    ):
        # Dataloaders and samplers
        self.tr_loader = data['tr_loader']      # Training DataLoader
        self.va_loader = data['va_loader']      # Validation DataLoader
        self.ev_loader_list = data['ev_loader_list']      # Evaluation DataLoader
        self.tt_loader_list = data['tt_loader_list']      # Test Time Evaluation DataLoader
        
        self.model = model
        self.discriminator = discriminator
        self.optim = optim
        self.optim_disc = optim_disc
        self.scheduler = scheduler
        self.scheduler_disc = scheduler_disc

        # loss weights
        self.loss = args.loss

        # logger
        self.logger = logger

        # dataset
        self.segment = args.segment
        self.n_fft = args.n_fft
        self.hop_size = args.hop_size
        self.win_size = args.win_size
        self.compress_factor = args.compress_factor
        self.stft_args = {
            "n_fft": args.n_fft,
            "hop_size": args.hop_size,
            "win_size": args.win_size,
            "compress_factor": args.compress_factor
        }
        self.input_type = args.model.input_type

        # Basic config
        self.device = device or torch.device(args.device)
        
        self.epochs = args.epochs
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        
        self.writer = None
        self.best_state = None
        self.best_loss = 0.0
        self.history = []
        self.log_dir = args.log_dir
        self.samples_dir = args.samples_dir
        self.num_prints = args.num_prints
        self.num_workers = args.num_workers
        self.args = args
        
        # Initialize or resume (checkpoint loading)
        self._reset()
    
    def _serialize(self):
        """ Save states checkpoint. """
        package = {}
        package['model'] = copy_state(self.model.state_dict())
        package['best_state'] = self.best_state
        package['discriminator'] = copy_state(self.discriminator.state_dict())
        package['optimizer'] = self.optim.state_dict()
        package['optimizer_disc'] = self.optim_disc.state_dict()
        package['scheduler'] = self.scheduler.state_dict() if self.scheduler is not None else None
        package['scheduler_disc'] = self.scheduler_disc.state_dict() if self.scheduler_disc is not None else None
        package['args'] = self.args
        package['history'] = self.history
        package['best_loss'] = self.best_loss
        # Write to a temporary file first
        tmp_path = "checkpoint.tmp"
        torch.save(package, tmp_path)
        os.rename(tmp_path, f"checkpoint.th")

        best_path = "best.tmp"
        torch.save(self.best_state, best_path)
        os.rename(best_path, "best.th")

    def _reset(self):
        """Load checkpoint if 'continue_from' is specified, or create a fresh writer if not."""
        if self.continue_from is not None:
            self.logger.info(f'Loading checkpoint model: {self.continue_from}')
            if not os.path.exists(self.continue_from):
                raise FileNotFoundError(f"Checkpoint directory {self.continue_from} not found.")
            
            # Attempt to copy the 'tensorbd' directory (TensorBoard logs) if it exists
            src_tb_dir = os.path.join(self.continue_from, 'tensorbd')
            dst_tb_dir = self.log_dir
            
            if os.path.exists(src_tb_dir):
                # If the previous tensorboard logs exist, we either copy them
                # to the new log dir or skip if it already exists.
                if not os.path.exists(dst_tb_dir):
                    shutil.copytree(src_tb_dir, dst_tb_dir)
                else:
                    # If the new log dir already exists, just issue a warning and do not overwrite
                    self.logger.warning(f"TensorBoard log dir {dst_tb_dir} already exists. Skipping copy.")
                # Initialize the SummaryWriter to continue logging in the (possibly copied) directory
                self.writer = SummaryWriter(log_dir=dst_tb_dir)
                        
            # loads the checkpoint file from disk
            ckpt_path = os.path.join(self.continue_from, 'checkpoint.th')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
            self.logger.info(f"Loading checkpoint from {ckpt_path}")
            package = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                            
            model_state = package['model']
            model_disc_state = package.get('discriminator', None)
            optim_state = package['optimizer']
            optim_disc_state = package.get('optimizer_disc', None)
            scheduler_state = package.get('scheduler', None)
            scheduler_disc_state = package.get('scheduler_disc', None)
            self.best_loss = package.get('best_loss', 0.0)
            self.best_state = package.get('best_state', None)
            self.history = package.get('history', [])
                    
            self.model.load_state_dict(model_state)
            self.optim.load_state_dict(optim_state)
            
            self.discriminator.load_state_dict(model_disc_state)
            self.optim_disc.load_state_dict(optim_disc_state)
            
            if self.scheduler is not None and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            if self.scheduler_disc is not None and scheduler_disc_state is not None:
                self.scheduler_disc.load_state_dict(scheduler_disc_state)
            
        else:
            # If there's no checkpoint to resume from, just create a fresh SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)


    def train(self):

        if self.history:
            self.logger.info("Replaying metrics from previous run")
            for epoch, metrics in enumerate(self.history):
                info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
                self.logger.info(f"Epoch {epoch + 1}: {info}")

        self.logger.info(f"Training for {self.epochs} epochs")

        for epoch in range(len(self.history), self.epochs):

            self.model.train()

            start = time.time()
            self.logger.info('-' * 70)
            self.logger.info("Training...")
            self.logger.info(f"Train | Epoch {epoch + 1} | Learning Rate {self.optim.param_groups[0]['lr']:.6f}")
            train_loss = self._run_one_step(epoch)

            self.logger.info(
                f"Train Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}"
            )

            self.model.eval()

            start = time.time()
            self.logger.info('-' * 70)
            self.logger.info('Validation...')
            with torch.no_grad():
                valid_loss = self._run_one_step(epoch, valid=True)
            
            self.logger.info(
                f"Valid Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}"
            )

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss}
            self.history.append(metrics)
            info = " | ".join(f"{k} {v:.5f}" for k, v in metrics.items())
            self.best_loss = min(self.best_loss, best_loss)
            self.logger.info('-' * 70)
            self.logger.info(f"Overall Summary | Epoch {epoch + 1} | {info}")
            
            if valid_loss == best_loss:
                self.logger.info(f'New best valid loss {valid_loss:.4f}')
                self.best_state = {'model': copy_state(self.model.state_dict())}

            self._serialize()
           
            if (epoch + 1) % self.eval_every == 0:
                self.logger.info('-' * 70)
                self.logger.info('Evaluating on the test set...')
                with swap_state(self.model, self.best_state['model']):
                    ev_metric = evaluate(
                        args=self.args, 
                        model=self.model, 
                        data_loader_list=self.ev_loader_list, 
                        logger=self.logger, 
                        epoch=epoch,
                        stft_args=self.stft_args)
                
                for snr, metric_item in ev_metric.items():
                    for k, v in metric_item.items():
                        self.writer.add_scalar(f"test/{snr}/{k}", v, epoch)
                
                

        self.logger.info("-" * 70)
        self.logger.info("Training Completed")
        self.logger.info("-" * 70)
        self.writer.close()

    def _run_one_step(self, epoch, valid=False):
        
        total_loss = 0.0
        data_loader = self.tr_loader if not valid else self.va_loader

        label = ["Train", "Valid"][valid]
        name = label + f" | Epoch {epoch + 1}"

        logprog = LogProgress(self.logger, data_loader, updates=self.num_prints, name=name)

        for i, data in enumerate(logprog):
            
            bcs, noisy_acs, clean_acs = data
            if self.input_type == "acs":
                input = mag_pha_stft(noisy_acs, **self.stft_args)[2].to(self.device)
            elif self.input_type == "bcs":
                input = mag_pha_stft(bcs, **self.stft_args)[2].to(self.device)
            elif self.input_type == "acs+bcs":
                input = mag_pha_stft(bcs, **self.stft_args)[2].to(self.device), \
                        mag_pha_stft(noisy_acs, **self.stft_args)[2].to(self.device)

            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_acs, **self.stft_args)
            clean_mag = clean_mag.to(self.device)
            clean_pha = clean_pha.to(self.device)
            clean_com = clean_com.to(self.device)
            one_labels = torch.ones(clean_mag.shape[0]).to(self.device)

            clean_mag_hat, clean_pha_hat, clean_com_hat = self.model(input)

            clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **self.stft_args)
            clean_mag_hat_con, clean_pha_hat_con, clean_com_hat_con = mag_pha_stft(clean_hat, **self.stft_args)


            if not valid:
                clean_list, clean_list_hat = list(clean_acs.cpu().numpy()), list(clean_hat.detach().cpu().numpy())
                batch_pesq_score = batch_pesq(clean_list, clean_list_hat, workers=self.num_workers)

                metric_r = self.discriminator(clean_mag.unsqueeze(1), clean_mag.unsqueeze(1))
                metric_g = self.discriminator(clean_mag.unsqueeze(1), clean_mag_hat_con.detach().unsqueeze(1))
                
                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

                if batch_pesq_score is not None:
                    loss_disc_g = F.mse_loss(batch_pesq_score.to(self.device), metric_g.flatten())
                else:
                    loss_disc_g = 0

                loss_disc = loss_disc_r + loss_disc_g
                
                self.optim_disc.zero_grad()
                loss_disc.backward()
                self.optim_disc.step()

                logprog.append(**{f"Disc_Loss": format(loss_disc.item(), "4.5f")})


            loss_magnitude = F.mse_loss(clean_mag, clean_mag_hat)
            loss_phase = phase_losses(clean_pha, clean_pha_hat)
            loss_complex = F.mse_loss(clean_com, clean_com_hat) * 2
            loss_consistency = F.mse_loss(clean_com_hat, clean_com_hat_con) * 2

            metric_g = self.discriminator(clean_mag.unsqueeze(1), clean_mag_hat_con.unsqueeze(1))
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

            loss_gen = loss_metric * self.loss.metric + \
                    loss_complex * self.loss.complex + \
                    loss_consistency * self.loss.consistency + \
                    loss_magnitude * self.loss.magnitude + \
                    loss_phase * self.loss.phase

            if not valid:
                self.optim.zero_grad()
                loss_gen.backward()
                self.optim.step()

            loss_dict = {
                "Magnitude_Loss": loss_magnitude.item(),
                "Phase_Loss": loss_phase.item(),
                "Complex_Loss": loss_complex.item(),
                "Consistency_Loss": loss_consistency.item(),
                "Metric_Loss": loss_metric.item(),
                "Gen_Loss": loss_gen.item()
            }

            for k, v in loss_dict.items():
                logprog.append(**{f"{k}": format(v, "4.5f")})
                if i % (self.num_prints * 10) == 0:
                    self.writer.add_scalar(f"{label}/{k}", v, epoch * len(data_loader) + i)

            total_loss += loss_gen.item()

        if not valid:
            if self.scheduler is not None:
                self.scheduler.step()
            if self.scheduler_disc is not None:
                self.scheduler_disc.step()

        return total_loss / len(data_loader)