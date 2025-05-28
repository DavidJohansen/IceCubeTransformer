print("Importing libraries")
import torch
import yaml

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from src.dataloader import make_dataloader_PMT, make_dataloader_PMT_multi

from src.model import regression_Transformer
from src.model import LitModel

from src.utils import assert_config_train

import torch.nn as nn

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Validate the config file
assert_config_train(config)

#==================================================================================================

device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')

#==================================================================================================
# Create the dataloaders
if config['input_data']['multi_dataset']:
    train_dataloader, val_dataloader = make_dataloader_PMT_multi(
        root_dir=config['input_data']['root_dir'],
        dataset_ids = config['input_data']['dataset_ids'],
        training_parts = config['input_data']['training_parts'],
        validation_parts = config['input_data']['validation_parts'],
        sample_weights = [4, 4, 1],
        batch_size=config['training_params']['batch_size'],
        num_workers=config['input_data']['num_workers'],
    )

else:
    train_dataloader, val_dataloader = make_dataloader_PMT(
        root_dir=config['input_data']['root_dir'],
        dataset_id = config['input_data']['dataset_ids'][0],
        training_parts = config['input_data']['training_parts'],
        validation_parts = config['input_data']['validation_parts'],
        batch_size=config['training_params']['batch_size'],
        num_workers=config['input_data']['num_workers'],
    )

# Print the number of batches in the training and validation sets

print(f"Number of batches in the training set: {len(train_dataloader)}")
print(f"Number of batches in the validation set: {len(val_dataloader)}")
#==================================================================================================
# Set checkpoint path

checkpoint_path = config['checkpoint_path']
if checkpoint_path != "None" and checkpoint_path is not None and checkpoint_path != "none":
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_path = checkpoint_path
else:
    checkpoint_path = None
    print("No checkpoint path provided. Starting training from scratch.")

#==================================================================================================
# Logging
if config['use_wandb']:
    wandb_logger = WandbLogger(
        project=config['wandb_params']['wandb_project'],
        name=config['run_name'],
        log_model=True,
    )
    wandb_logger.log_hyperparams(config)

if config['use_tensorboard']:
    tensorboard_logger = TensorBoardLogger(
        save_dir=config['tensorboard_params']['log_dir'],
        name=config['run_name'],
    )

if config['use_wandb'] and config['use_tensorboard']:
    logger = [wandb_logger, tensorboard_logger]
elif config['use_wandb']:
    logger = wandb_logger
elif config['use_tensorboard']:
    logger = tensorboard_logger
else:
    logger = None
    print("No logger specified. Skipping logging.")

#==================================================================================================
# Define the model













model = regression_Transformer(
    embedding_dim = config['model_params']['embedding_dim'], 
    n_layers = config['model_params']['n_layers'], 
    n_heads = config['model_params']['n_heads'], 
    input_dim = config['model_params']['feature_dim'], 
    seq_dim = config['input_data']['seq_dim'], 
    dropout = config['model_params']['dropout'], 
    output_dim = config['model_params']['output_dim'], 
).to(device)

#==================================================================================================
# Define optimizer based on config
optimizer_name = config['training_params']['optimizer']

eps = config['training_params']['adam_eps']
# convert eps to float if it's a string
if isinstance(eps, str):
    eps = float(eps)

if optimizer_name == "AdamW":
    optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config['training_params']['learning_rate'],
                    betas=(config['training_params']['beta1'], config['training_params']['beta2']), 
                    eps=eps,
                    weight_decay=config['training_params']['weight_decay'],
                    amsgrad=True, 
                )
elif optimizer_name == "Adam":
     optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config['training_params']['learning_rate'],
                    betas=(config['training_params']['beta1'], config['training_params']['beta2']), 
                    eps=eps, 
                    weight_decay=config['training_params']['weight_decay'],
                )
# Add other optimizers like SGD if needed
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

#==================================================================================================
# Define scheduler based on config
if config['use_one_cycle_lr']:
    one_cycle_params = config['one_cycle_lr_params']
    scheduler_config = {
        'scheduler': torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=one_cycle_params['max_lr'], # Updated
            epochs=config['training_params']['n_epochs'], # Updated
            steps_per_epoch=len(train_dataloader),
            pct_start=one_cycle_params['pct_start'], # Updated
            anneal_strategy=one_cycle_params['anneal_strategy'], # Updated
            div_factor=one_cycle_params['div_factor'], # Updated
            final_div_factor=one_cycle_params['final_div_factor'] # Updated
        ),
        'interval': 'step',
        'frequency': 1,
    }
    optimizer_list = [optimizer]
    scheduler_list = [scheduler_config]
else:
    # No scheduler or configure a different one if needed
    optimizer_list = [optimizer]
    scheduler_list = []

optimizer = optimizer_list, scheduler_list

#==================================================================================================
# Define the Lightning model - Pass optimizer/scheduler lists
import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml

import numpy as np
from src.model import FeedForward
from src.loss import dir_3vec_loss, MSE_loss, VonMisesFisherLoss3D, opening_angle_loss, Simon_loss

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# set loss func based on the config file but not the string
if config['training_params']['loss_function'] == 'dir_3vec_loss':
    loss_func = dir_3vec_loss
elif config['training_params']['loss_function'] == 'MSE_loss':
    loss_func = MSE_loss
elif config['training_params']['loss_function'] == 'VonMisesFisherLoss3D':
    loss_func = VonMisesFisherLoss3D
elif config['training_params']['loss_function'] == 'opening_angle_loss':
    loss_func = opening_angle_loss
elif config['training_params']['loss_function'] == 'Simon_loss':
    loss_func = Simon_loss


class LitModel2(pl.LightningModule):
    def __init__(
            self, 
            optimizer,
            train_dataset, 
            val_dataset,
            batch_size=16,
            ):
        super(LitModel, self).__init__()
        self.model = FeedForward()
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

        # Store the training and validation losses
        self.train_losses = []
        self.val_losses = []

        self.train_opening_angles = []
        self.val_opening_angles = []

    def forward(self, x, event_lengths=None):
        return self.model(x, train_dataset.shape[0])

    def training_step(self, batch, batch_idx):
        x, target, event_lengths = batch[0], batch[1], batch[2]
        y_pred, loss = self.model(x, target=target, event_lengths=event_lengths)
        #mean_loss = torch.mean(loss)
        self.train_losses.append(loss.item())

        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, prog_bar=False, logger=True, sync_dist=True)

        if batch_idx % 100 == 0:
            # Print y_pred and target for the first 5 events in the batch
            print("\n")
            print("y_pred: ", y_pred[:5])
            print("target: ", target[:5])

            pred_x = y_pred[:5, 0].detach().cpu().to(torch.float32).numpy()
            pred_y = y_pred[:5, 1].detach().cpu().to(torch.float32).numpy()
            pred_z = y_pred[:5, 2].detach().cpu().to(torch.float32).numpy()

            target_x = target[:5, 0].detach().cpu().to(torch.float32).numpy()
            target_y = target[:5, 1].detach().cpu().to(torch.float32).numpy()
            target_z = target[:5, 2].detach().cpu().to(torch.float32).numpy()

            opening_angle = np.arccos((pred_x * target_x + pred_y * target_y + pred_z * target_z) / (np.sqrt(pred_x**2 + pred_y**2 + pred_z**2) * np.sqrt(target_x**2 + target_y**2 + target_z**2))) * 180 / np.pi
            print("Opening angle (deg): ", opening_angle)

            self.train_opening_angles.append(opening_angle)

        
        self.log('train_loss', loss, prog_bar=True, on_step=True, logger=True, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
        # Log the median training loss at the end of each epoch
        median_train_loss = torch.tensor(self.train_losses).median().item()
        self.log('median_train_loss', median_train_loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        mean_train_opening_angle = np.mean(self.train_opening_angles)
        median_train_opening_angle = np.median(self.train_opening_angles)
        self.log('mean_train_opening_angle', mean_train_opening_angle, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('median_train_opening_angle', median_train_opening_angle, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        self.train_losses = []
        self.train_opening_angles = []
        
    def validation_step(self, batch, batch_idx):
        x, target, event_lengths = batch[0], batch[1], batch[2]
        y_pred, loss = self.model(x, target=target, event_lengths=event_lengths)
        #loss = torch.mean(loss)
        self.val_losses.append(loss.item())

        if batch_idx % 100 == 0:
            # Print y_pred and target for the first 5 events in the batch
            print("\n")
            print("y_pred: ", y_pred[:5])
            print("target: ", target[:5])

        if batch_idx % 10 == 0:

            pred_x = y_pred[:5, 0].detach().cpu().to(torch.float32).numpy()
            pred_y = y_pred[:5, 1].detach().cpu().to(torch.float32).numpy()
            pred_z = y_pred[:5, 2].detach().cpu().to(torch.float32).numpy()

            target_x = target[:5, 0].detach().cpu().to(torch.float32).numpy()
            target_y = target[:5, 1].detach().cpu().to(torch.float32).numpy()
            target_z = target[:5, 2].detach().cpu().to(torch.float32).numpy()

            opening_angle = np.arccos((pred_x * target_x + pred_y * target_y + pred_z * target_z) / (np.sqrt(pred_x**2 + pred_y**2 + pred_z**2) * np.sqrt(target_x**2 + target_y**2 + target_z**2))) * 180 / np.pi
            print("Opening angle (deg): ", opening_angle)

            self.val_opening_angles.append(opening_angle)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

    def on_validation_epoch_end(self):
        # Log the median validation loss at the end of each epoch
        median_val_loss = torch.tensor(self.val_losses).median().item()
        self.log('median_val_loss', median_val_loss, prog_bar=True, on_epoch=True, logger=True)

        mean_val_opening_angle = np.mean(self.val_opening_angles)
        median_val_opening_angle = np.median(self.val_opening_angles)
        self.log('mean_val_opening_angle', mean_val_opening_angle, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('median_val_opening_angle', median_val_opening_angle, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        self.val_losses = []
        self.val_opening_angles = []

    def predict_step(self, batch, batch_idx):
        x, target, event_lengths = batch[0], batch[1], batch[2]
        y_pred, _ = self.model(x, event_lengths=event_lengths)
        return {'y_pred': y_pred, 'target': target}

    def configure_optimizers(self):
        return self.optimizer


lit_model = SimpleMLP(
    optimizer,
    train_dataloader, # Pass dataloader directly if LitModel uses it internally
    val_dataloader,   # Pass dataloader directly if LitModel uses it internally
    batch_size=config['training_params']['batch_size'], # Updated
)

#==================================================================================================
# Define the trainer
 
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=config['training_params']['patience'], 
        verbose=True,
        mode='min' 
    ),
    ModelCheckpoint(
        dirpath=tensorboard_logger.log_dir,
        filename='transformer-{epoch:02d}-{val_loss:.4f}', # Include metric in filename
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        mode='min',
    ),
    TQDMProgressBar(
        refresh_rate=100,
    ),
            ]

trainer = LitModel2(
    accelerator= 'cpu', #'gpu' if 'cuda' in config['training_params']['device'] else 'cpu', 
    devices = 1, # [int(config['training_params']['device'].split(':')[-1])] if 'cuda' in config['training_params']['device'] else 1, 
    max_epochs=config['training_params']['n_epochs'], 
    log_every_n_steps=config['log_every_n_steps'], 
    logger=logger,
    callbacks=callbacks,
)

#==================================================================================================
# Train the model
if checkpoint_path is not None:
    trainer.fit(lit_model, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
else:
    trainer.fit(lit_model, train_dataloader, val_dataloader)
