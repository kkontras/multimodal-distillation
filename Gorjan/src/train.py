import logging
import random
from os.path import join as pjoin

import torch
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from yacs.config import CfgNode

from modelling.datasets import dataset_factory
from modelling.losses import LossesModule
from modelling.models import model_factory
from utils.data_utils import separator
from utils.evaluation import evaluators_factory
from utils.setup import train_setup
from utils.train_utils import get_linear_schedule_with_warmup
import wandb
import os
import numpy as np

# torch._dynamo.config.suppress_errors = True

def load_prev_run(cfg, model, best_model, optimizer, scheduler, logs):

    if cfg.LOAD_EXPERIMENT:
        checkpoint = torch.load(os.path.join(str(cfg.EXPERIMENT_PATH), "general_checkpoint.pth.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        best_model.load_state_dict(checkpoint["best_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logs = checkpoint['logs']
        logs["current_epoch"] +=1
        print("Loaded experiment from {}".format(cfg.EXPERIMENT_PATH))
        logging.info(f"Loaded experiment from {cfg.EXPERIMENT_PATH}")

        # print(logs["train_logs"].keys())
        # print(logs["val_logs"].keys())

        for step in logs["train_logs"]:
            if step in logs["val_logs"]:
                wandb.log({"train_metrics": logs["train_logs"][step], "val_metrics": logs["val_logs"][step]}, step=step)
            else:
                wandb.log({"train_metrics": logs["train_logs"][step]}, step=step)



    return model, best_model, optimizer, scheduler, logs

def save(cfg, model, best_model, optimizer, scheduler, logs):
    save_dict = {}
    savior = {}
    savior["model_state_dict"] = model.state_dict()
    savior["best_model_state_dict"] = best_model.state_dict()
    savior["optimizer_state_dict"] = optimizer.state_dict()
    savior["scheduler_state_dict"] = scheduler.state_dict()
    savior["logs"] = logs
    savior["configs"] = cfg

    save_dict.update(savior)
    try:
        # print(cfg.EXPERIMENT_PATH)
        torch.save(best_model.state_dict(), os.path.join(str(cfg.EXPERIMENT_PATH), "model_checkpoint.pt"))
        torch.save(save_dict, os.path.join(str(cfg.EXPERIMENT_PATH), "general_checkpoint.pth.tar"))
        torch.save(save_dict, os.path.join(str(cfg.EXPERIMENT_PATH), "general_checkpoint_cp.pth.tar"))

    except:
        raise Exception("Problem in model saving")
    return True

def init_logs(cfg):
    logs = {"current_epoch": 0,
            "current_step": 0,
            "steps_no_improve": 0,
            "saved_step": 0,
            "train_logs": {},
            "val_logs": {},
            "test_logs": {},
            "best_logs": {"val_loss": {"total": 100}, "val_acc": {"combined": 0}},
            "seed": cfg.SEED}

    return logs

def deterministic(seed):
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_dataloaders(cfg, accelerator):

    if cfg.LOG_TO_FILE:
        if accelerator.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                filename=pjoin(cfg.EXPERIMENT_PATH, "experiment_log.log"),
                filemode="w",
            )
    else:
        logging.basicConfig(level=logging.INFO)
    if accelerator.is_main_process:
        logging.info(separator)
        logging.info(f"The config file is:\n {cfg}")
        logging.info(separator)
    # Prepare datasets
    if accelerator.is_main_process:
        logging.info("Preparing datasets...")
    # Prepare train dataset
    train_dataset = dataset_factory[cfg.DATASET_TYPE](cfg, train=True)
    # Prepare validation dataset
    val_dataset = dataset_factory[cfg.DATASET_TYPE](cfg, train=False)
    num_training_samples = len(train_dataset)
    if cfg.VAL_SUBSET:
        val_indices = random.sample(range(len(val_dataset)), cfg.VAL_SUBSET)
        val_dataset = Subset(val_dataset, val_indices)
    num_validation_samples = len(val_dataset)
    if accelerator.is_main_process:
        logging.info(f"Training on {num_training_samples}")
        logging.info(f"Validating on {num_validation_samples}")
    # Prepare loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.NUM_WORKERS else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.NUM_WORKERS else False,
    )
    return train_loader, val_loader, num_training_samples, num_validation_samples
def train(cfg: CfgNode, accelerator: Accelerator):


    wandb_run = wandb.init(reinit=True, project="balance", config=cfg,
                                dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/wandb",
                                name=cfg.EXPERIMENT_PATH)

    deterministic(cfg.SEED)

    train_loader, val_loader, num_training_samples, num_validation_samples = get_dataloaders(cfg, accelerator)

    if accelerator.is_main_process: logging.info("Preparing model...")
    # Prepare model
    model = model_factory[cfg.MODEL_NAME](cfg)
    best_model = model_factory[cfg.MODEL_NAME](cfg)
    best_model.load_state_dict(model.state_dict())
    wandb.watch(model, log_freq=100)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    num_batches = num_training_samples // cfg.BATCH_SIZE
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.WARMUP_EPOCHS * num_batches,
        num_training_steps=cfg.EPOCHS * num_batches,
    )

    logs = init_logs(cfg)
    model, best_model, optimizer, scheduler, logs = load_prev_run(cfg, model, best_model, optimizer, scheduler, logs)

    train_evaluator = evaluators_factory[cfg.VAL_DATASET_NAME](num_validation_samples, cfg)
    evaluator = evaluators_factory[cfg.VAL_DATASET_NAME](num_validation_samples, cfg)
    # Accelerate
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    # Create loss
    criterion = LossesModule(cfg)
    if accelerator.is_main_process:
        logging.info("Starting training...")
    # logs["current_step"] = logs["current_epoch"] * num_training_samples
    for logs["current_epoch"] in range(logs["current_epoch"], cfg.EPOCHS):
        # Training loop
        model.train(True)
        with tqdm(total=len(train_loader), disable=not accelerator.is_main_process) as pbar:
            for step, batch in enumerate(train_loader):
                train_evaluator.reset()
                optimizer.zero_grad()

                model_output = model(batch)
                loss = criterion(model_output, batch)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), cfg.CLIP_VAL)
                optimizer.step()
                scheduler.step()

                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
                for param_group in optimizer.param_groups: lr = param_group['lr']

                all_outputs = accelerator.gather(model_output)
                all_labels = accelerator.gather(batch["labels"])
                # Reshape outputs and put on cpu
                for key in all_outputs.keys():
                    num_classes = all_outputs[key].size(-1)
                    # Reshape
                    all_outputs[key] = all_outputs[key].reshape(
                        -1, cfg.NUM_TEST_CLIPS * cfg.NUM_TEST_CROPS, num_classes
                    )
                    # Move on CPU
                    all_outputs[key] = all_outputs[key].cpu()
                # Put labels on cpu
                for key in all_labels.keys():
                    all_labels[key] = all_labels[key].cpu()
                # Pass to evaluator
                train_evaluator.process(all_outputs, all_labels)
                train_metrics = train_evaluator.evaluate()
                train_metrics["loss"] = loss.item()
                train_metrics["lr"] = lr
                # wandb.log({"lr": lr, "train_loss": loss.item()}, step= epoch*num_training_samples + step)
                wandb.log({"train_metrics": train_metrics}, step=logs["current_step"])
                logs["train_logs"][logs["current_step"]] = train_metrics
                logs["current_step"] += 1
                # if step == 50:
                #     break

        # Validation loop
        model.train(False)
        evaluator.reset()
        for step, batch in tqdm(enumerate(val_loader), disable=not accelerator.is_main_process):
            with torch.no_grad():
                # Obtain outputs: [b * n_clips, n_actions]
                model_output = model(batch)
                all_outputs = accelerator.gather(model_output)
                all_labels = accelerator.gather(batch["labels"])
                # Reshape outputs and put on cpu
                for key in all_outputs.keys():
                    num_classes = all_outputs[key].size(-1)
                    # Reshape
                    all_outputs[key] = all_outputs[key].reshape(
                        -1, cfg.NUM_TEST_CLIPS * cfg.NUM_TEST_CROPS, num_classes
                    )
                    # Move on CPU
                    all_outputs[key] = all_outputs[key].cpu()
                # Put labels on cpu
                for key in all_labels.keys():
                    all_labels[key] = all_labels[key].cpu()
            # Pass to evaluator
            evaluator.process(all_outputs, all_labels)
            # if step == 50:
            #     break

        # Evaluate & save model
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metrics = evaluator.evaluate()
            wandb.log({"val_metrics": metrics}, step=logs["current_step"])
            logs["val_logs"][logs["current_step"]] = metrics

            if evaluator.is_best():
                logging.info(separator)
                logging.info(f"Found new best on epoch {logs['current_epoch']+1}!")
                logging.info(separator)
                print("Found new best on epoch {}!".format(logs['current_epoch']+1))
                logs["best_logs"] = metrics

                unwrapped_model = accelerator.unwrap_model(model)

                best_model.load_state_dict(unwrapped_model.state_dict())


            for m in metrics.keys():
                logging.info(f"{m}: {metrics[m]}")
                print("{}: {}".format(m, metrics[m]))

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_best_model = accelerator.unwrap_model(best_model)

            save(cfg, unwrapped_model, unwrapped_best_model, optimizer, scheduler, logs)

    wandb_run.finish()


def main():
    cfg, accelerator = train_setup("Trains an action recognition model.")
    train(cfg, accelerator)


if __name__ == "__main__":
    main()
