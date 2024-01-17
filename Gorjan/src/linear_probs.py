import argparse
import logging
from collections import OrderedDict
from os.path import join as pjoin

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from modelling.datasets import dataset_factory
from modelling.models import model_factory
from utils.calibration import CalibrationEvaluator
from utils.data_utils import separator
from utils.evaluation import evaluators_factory
from utils.setup import get_cfg_defaults
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import numpy as np

def unwrap_compiled_checkpoint(checkpoint):
    # If Pytorch 2.0, checkpoint is wrapped with _orig_mod, so we remove it
    new_checkpoint = OrderedDict()
    for key in checkpoint.keys():
        new_key = key[10:] if key.startswith("_orig_mod") else key
        new_checkpoint[new_key] = checkpoint[key]

    return new_checkpoint


@torch.no_grad()
def linear_probs(cfg: CfgNode):
    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator()
    # Prepare datasets
    if accelerator.is_main_process:
        logging.info("Preparing datasets...")
    # Prepare validation dataset
    if accelerator.is_main_process:
        logging.info(separator)
        logging.info(f"The config is:\n{cfg}")
        logging.info(separator)
    train_dataset = dataset_factory[cfg.DATASET_TYPE](cfg, train=True)
    val_dataset = dataset_factory[cfg.DATASET_TYPE](cfg, train=False)
    if accelerator.is_main_process:
        logging.info(f"Validating on {len(val_dataset)}")
    # Prepare loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.NUM_WORKERS else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True if cfg.NUM_WORKERS else False,
    )
    if accelerator.is_main_process:
        logging.info("Preparing model...")
    # Prepare model
    model = model_factory[cfg.MODEL_NAME](cfg)
    checkpoint = torch.load(
        pjoin(cfg.EXPERIMENT_PATH, "model_checkpoint.pt"), map_location="cpu"
    )
    checkpoint = unwrap_compiled_checkpoint(checkpoint)
    model.load_state_dict(checkpoint)

    if accelerator.is_main_process:
        logging.info("Starting inference...")
    # Accelerate
    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)
    model.train(False)

    features = defaultdict(list)
    targets = []

    for step, batch in tqdm(enumerate(train_loader), disable=not accelerator.is_main_process):
        # Obtain outputs: [b * n_clips, n_actions]
        model_output = model(batch, return_features=True)
        # Gather
        all_outputs = accelerator.gather(model_output)
        all_labels = accelerator.gather(batch["labels"])

        # print(all_outputs.keys())
        for i in all_outputs["features"]:
            features[i].append(all_outputs["features"][i])
        targets.append(all_labels["ACTION"])

        # if step == 200:
        #     break

        #TODO: Keep features of each modality
        # for each modality:
        #   learn the linear probs and pass them to the next loop to evaluate them.

        # all_labels = accelerator.gather(batch["labels"])
        # # Reshape outputs and put on cpu
        # for key in all_outputs.keys():
        #     num_classes = all_outputs[k
        #     ey].size(-1)
        #     # Reshape
        #     all_outputs[key] = all_outputs[key].reshape(
        #         -1, cfg.NUM_TEST_CLIPS * cfg.NUM_TEST_CROPS, num_classes
        #     )
        #     # Move on CPU
        #     all_outputs[key] = all_outputs[key].cpu()
        # # Put labels on cpu
        # for key in all_labels.keys():
        #     all_labels[key] = all_labels[key].cpu()
        # # Evaluate
        # evaluator.process(all_outputs, all_labels)
        # calibration_evaluator.process(all_outputs, all_labels)
    features = {i: torch.cat(features[i], dim=0) for i in features}
    targets = torch.cat(targets, dim=0)
    clf = {i: LogisticRegression(random_state=0, max_iter=10000, multi_class="multinomial").fit(features[i].cpu().numpy(), targets.cpu().numpy()) for i in features}

    # print(features["video"].shape)

    evaluator = evaluators_factory[cfg.VAL_DATASET_NAME](len(val_dataset), cfg)
    calibration_evaluator = CalibrationEvaluator(cfg)
    evaluator.reset()

    unimodal_eval = {i: evaluators_factory[cfg.VAL_DATASET_NAME](len(val_dataset), cfg) for i in features}
    unimodal_cal_eval = {i: CalibrationEvaluator(cfg) for i in features}
    for i in features: unimodal_eval[i].reset()

    # Inference
    for batch in tqdm(val_loader, disable=not accelerator.is_main_process):
        # Obtain outputs: [b * n_clips, n_actions]
        model_output = model(batch, return_features=True)
        # Gather
        all_outputs = accelerator.gather(model_output)
        all_labels = accelerator.gather(batch["labels"])

        num_classes = all_outputs["ACTION"].size(-1)
        all_outputs["ACTION"] = all_outputs["ACTION"].reshape(-1, cfg.NUM_TEST_CLIPS * cfg.NUM_TEST_CROPS, num_classes)
        all_outputs["ACTION"] = all_outputs["ACTION"].cpu()
        all_labels["ACTION"] = all_labels["ACTION"].cpu()

        # Evaluate
        evaluator.process(all_outputs, all_labels)
        calibration_evaluator.process(all_outputs, all_labels)

        preds = {}
        for i in all_outputs["features"]:
            preds[i] = clf[i].predict_proba(all_outputs["features"][i].cpu().numpy())
            preds[i] = torch.from_numpy(np.pad(preds[i], ((0, 0), (0, 174 - preds[i].shape[1]))))

        for i in features:
            print(preds[i].shape)
            unimodal_eval[i].process({"ACTION":preds[i].reshape(-1, cfg.NUM_TEST_CLIPS * cfg.NUM_TEST_CROPS, num_classes)}, all_labels)
            unimodal_cal_eval[i].process({"ACTION":preds[i].reshape(-1, cfg.NUM_TEST_CLIPS * cfg.NUM_TEST_CROPS, num_classes)}, all_labels)


        break
    # Metrics
    if accelerator.is_main_process:
        metrics = evaluator.evaluate_verbose()
        for m in metrics.keys():
            logging.info(f"{m}: {metrics[m]}")
        # Calibration Metrics
        calibration_metrics = calibration_evaluator.evaluate()
        logging.info("=============== Calibration Metrics (ECE) ===============")
        for m in calibration_metrics.keys():
            logging.info(f"{m}: {calibration_metrics[m]}")

        for i in unimodal_eval:

            metrics = unimodal_eval[i].evaluate_verbose()
            for m in metrics.keys():
                logging.info(f"{i}-{m}: {metrics[m]}")
            # Calibration Metrics
            calibration_metrics = unimodal_cal_eval[i].evaluate()
            logging.info("=============== Calibration Metrics (ECE) ===============")
            for m in calibration_metrics.keys():
                logging.info(f"{i}-{m}: {calibration_metrics[m]}")


def main():
    parser = argparse.ArgumentParser(description="Inference.")
    parser.add_argument(
        "--experiment_path", required=True, help="Path to the experiment."
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(pjoin(args.experiment_path, "config.yaml"))
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Set backbone_model_path to None as not important
    cfg.BACKBONE_MODEL_PATH = None
    cfg.BATCH_SIZE = 32
    # Freeze the config
    cfg.freeze()
    linear_probs(cfg)


if __name__ == "__main__":
    main()
