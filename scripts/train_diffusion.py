# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler

"""Script used to train a ATISS with multi-GPU support using Accelerator."""
import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# Accelerator for multi GPU support
from accelerate import Accelerator

from training_utils import id_generator, save_experiment_params, load_config, yield_forever, load_checkpoints, save_checkpoints
from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import build_network, optimizer_factory, schedule_factory, adjust_learning_rate
from scene_synthesis.stats_logger import StatsLogger, WandB


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Initialize Accelerator for multi GPU support.
    accelerator = Accelerator()

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    # Use accelerator's device
    device = accelerator.device
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Save the parameters of this run to a file
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_directory))

    # Parse the config file
    config = load_config(args.config_file)

    # Load training dataset and save bounds
    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"])
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    np.savez(
        path_to_bounds,
        sizes=train_dataset.bounds["sizes"],
        translations=train_dataset.bounds["translations"],
        angles=train_dataset.bounds["angles"],
        # add objfeats
        objfeats=train_dataset.bounds["objfeats"],
    )
    print("Saved the dataset bounds in {}".format(path_to_bounds))

    # Load validation dataset using saved bounds
    validation_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=path_to_bounds,
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )

    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    print("Loaded {} training scenes with {} object types".format(
        len(train_dataset), train_dataset.n_object_types)
    )
    print("Training set has {} bounds".format(train_dataset.bounds))

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_processes,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False
    )
    print("Loaded {} validation scenes with {} object types".format(
        len(validation_dataset), validation_dataset.n_object_types)
    )
    print("Validation set has {} bounds".format(validation_dataset.bounds))

    # Ensure train and validation datasets have the same object categories
    assert train_dataset.object_types == validation_dataset.object_types

    # Build the network and get batch processing functions
    network, train_on_batch, validate_on_batch = build_network(
        train_dataset.feature_size, train_dataset.n_classes,
        config, args.weight_file, device=device
    )
    n_all_params = int(sum([np.prod(p.size()) for p in network.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, network.parameters())]))
    print(f"Number of parameters in {network.__class__.__name__}:  {n_trainable_params} / {n_all_params}")

    # Build an optimizer
    optimizer = optimizer_factory(config["training"], filter(lambda p: p.requires_grad, network.parameters()))

    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, experiment_directory, args, device)
    # Load the learning rate scheduler 
    lr_scheduler = schedule_factory(config["training"])

    # Wrap the model, optimizer, and data loaders with Accelerator
    network, optimizer, train_loader, val_loader = accelerator.prepare(
        network, optimizer, train_loader, val_loader
    )

    # Load checkpoints (unwrap the network for correct state dict loading)
    load_checkpoints(accelerator.unwrap_model(network), optimizer, experiment_directory, args, accelerator.device)

    # Initialize WandB logger if requested and if main process
    if args.with_wandb_logger and accelerator.is_main_process:
        WandB.instance().init(
            config,
            model=accelerator.unwrap_model(network),
            project=config["logger"].get("project", "autoregressive_transformer"),
            name=experiment_tag,
            watch=False,
            log_frequency=10
        )

    # Initialize StatsLogger in main process only to avoid duplicate logs
    if accelerator.is_main_process:
        StatsLogger.instance().add_output_file(open(os.path.join(experiment_directory, "stats.txt"), "w"))

    epochs = config["training"].get("epochs", 150)
    # steps_per_epoch is not used since we iterate over the DataLoader directly.
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    # Training loop
    for i in range(args.continue_from_epoch, epochs):
        adjust_learning_rate(lr_scheduler, optimizer, i)

        network.train()
        for b, sample in enumerate(train_loader):
            # Accelerator automatically handles device placement for batch data.
            batch_loss = train_on_batch(network, optimizer, sample, config)
            accelerator.backward(batch_loss)

            # Aggregate loss from all processes and compute average loss
            avg_loss = accelerator.gather(batch_loss).mean().item()
            if accelerator.is_main_process:
                StatsLogger.instance().print_progress(i+1, b+1, avg_loss)

            # gradient clipping과 optimizer.step()은 backward 호출 후에 진행합니다.
            grad_norm = clip_grad_norm_(network.parameters(), config["training"]["max_grad_norm"])
            StatsLogger.instance()["gradnorm"].value = grad_norm.item()
            StatsLogger.instance()["lr"].value = optimizer.param_groups[0]['lr']
            optimizer.step()

        # Checkpoint saving (only main process)
        if accelerator.is_main_process and (i % save_every) == 0:
            save_checkpoints(
                i,
                accelerator.unwrap_model(network),
                optimizer,
                experiment_directory,
            )
            StatsLogger.instance().clear()

        # Validation loop every val_every epochs (only main process prints logs)
        if i % val_every == 0 and i > 0:
            if accelerator.is_main_process:
                print("====> Validation Epoch ====>")
            network.eval()
            for b, sample in enumerate(val_loader):
                batch_loss = validate_on_batch(network, sample, config)
                avg_loss_val = accelerator.gather(batch_loss).mean().item()
                if accelerator.is_main_process:
                    StatsLogger.instance().print_progress(-1, b+1, avg_loss_val)
            if accelerator.is_main_process:
                StatsLogger.instance().clear()
                print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main(sys.argv[1:])
