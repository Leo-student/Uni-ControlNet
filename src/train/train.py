import sys
if './' not in sys.path:
	sys.path.append('./')
	
from omegaconf import OmegaConf
import argparse

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./log_local/lightning_logs/version_11/checkpoints')

from ldm.util import instantiate_from_config
from models.util import load_state_dict
from models.logger import ImageLogger

import multiprocessing

parser = argparse.ArgumentParser(description='Uni-ControlNet Training')
parser.add_argument('--config-path', type=str, default='./configs/local_v15.yaml')
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--training-steps', type=int, default=1e3)
parser.add_argument('--resume-path', type=str, default='./ckpt/init_local.ckpt')
parser.add_argument('--logdir', type=str, default='./log_local/')
parser.add_argument('--log-freq', type=int, default=50)
parser.add_argument('--sd-locked', type=bool, default=True)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--gpus', type=int, default=-1)
args = parser.parse_args()


def main():

    config_path    = args.config_path
    learning_rate  = args.learning_rate
    batch_size     = args.batch_size
    training_steps = args.training_steps
    resume_path    = args.resume_path
    default_logdir = args.logdir
    logger_freq    = args.log_freq
    sd_locked      = args.sd_locked
    num_workers    = args.num_workers
    gpus           = args.gpus
 
    print(f'config_path    {config_path   }')
    print(f'learning_rate  {learning_rate }')
    print(f'batch_size     {batch_size    }')
    print(f'training_steps {training_steps}')
    print(f'resume_path    {resume_path   }')
    print(f'default_logdir {default_logdir}')
    print(f'logger_freq    {logger_freq   }')
    print(f'sd_locked      {sd_locked     }')
    print(f'num_workers    {num_workers   }')
    print(f'gpus           {gpus          }')
    
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config['model'])

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked

    dataset = instantiate_from_config(config['data'])
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq)
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=logger_freq,
    )
        
    trainer = pl.Trainer(
        gpus=gpus,
        callbacks=[logger, checkpoint_callback], 
        default_root_dir=default_logdir,
        max_steps=training_steps,
    )
    trainer.fit(model,
        dataloader, 
    )


# if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
main()