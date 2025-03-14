# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for generating scenes using a previously trained model."""
import argparse
import logging
import os
import sys

import numpy as np
import torch

from training_utils import load_config

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network

from tqdm import tqdm

def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()

def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default="../demo/floor_plan_texture_images",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_sequences",
        default=10,
        type=int,
        help="The number of sequences to be generated"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
    )
    parser.add_argument(
        "--render_top2down",
        action="store_true",
        help="Perform top2down orthographic rendering"
    )
    parser.add_argument(
        "--without_floor",
        action="store_true",
        help="if remove the floor plane"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="if remove the texture"
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="if save mesh"
    )
    parser.add_argument(
        "--mesh_format",
        type=str,
        default=".ply",
        help="mesh format "
    )
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        help="if clip_denoised"
    )
    #
    parser.add_argument(
        "--retrive_objfeats",
        action="store_true",
        help="if retrive most similar objectfeats"
    )
    parser.add_argument(
        "--fix_order",
        action="store_true",
        help="if use fix order"
    )
    parser.add_argument(
        "--compute_intersec",
        action="store_true",
        help="if remove the texture"
    )
    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    ########## make it for evaluation
    if 'text' in config["data"]["encoding_type"]:
        if 'textfix' not in config["data"]["encoding_type"]:
            config["data"]["encoding_type"] = config["data"]["encoding_type"].replace('text', 'textfix')

    if "no_prm" not in config["data"]["encoding_type"]:
        print('NO PERM AUG in test')
        config["data"]["encoding_type"] = config["data"]["encoding_type"] + "_no_prm"
    print('encoding type :', config["data"]["encoding_type"])
    ####### 

    raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"])
    )

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"])
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )
    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    print('init scene top2donw')
    given_scene_id = None
    if args.scene_id:
        for i, di in enumerate(raw_dataset):
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i       

    classes = np.array(dataset.class_labels)
    print('class labels:', classes, len(classes))
    for i in tqdm(range(len(dataset))):
        scene_idx = i
        current_scene = raw_dataset[scene_idx]
        samples = dataset[scene_idx]
        print("{} / {}: Using the {} floor plan of scene {}".format(
            i, args.n_sequences, scene_idx, current_scene.scene_id)
        )
        room_mask = torch.from_numpy(
            np.transpose(current_scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        )

        bbox_params = network.generate_layout(
                room_mask=room_mask.to(device),
                num_points=config["network"]["sample_num_points"],
                point_dim=config["network"]["point_dim"],
                #text=torch.from_numpy(samples['desc_emb'])[None, :].to(device) if 'desc_emb' in samples.keys() else None, # glove embedding
                text=samples['description'] if 'description' in samples.keys() else None,  # bert 
                device=device,
                clip_denoised=args.clip_denoised,
                batch_seeds=torch.arange(i, i+1),
        )

        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).cpu().numpy()
        path_to_npy = os.path.join(
            args.output_directory,
            "{}.npy".format(current_scene.scene_id)
        )
        np.save(path_to_npy, bbox_params_t)  # bbox_params_t를 npy 파일로 저장
        print('Generated bbox:', bbox_params_t.shape)

        if "description" in samples.keys():
            path_to_texts = os.path.join(
                args.output_directory,
                "{}.txt".format(current_scene.scene_id)
            )
            print('the length of samples[description]: {:d}'.format( len(samples['description']) ) )
            print('text description {}'.format( samples['description']) )
            open(path_to_texts, 'w').write( ''.join(samples['description']) )


if __name__ == "__main__":
    main(sys.argv[1:])
