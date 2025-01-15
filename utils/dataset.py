import math
import os.path
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from glob import glob
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import torch
import yaml
import glob

# Visualization for Debugging
from matplotlib import pyplot as plt
from pandas import read_csv
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from utils.augmentations import *
from utils.torch_utils import *
from utils.track_utils import PoseInterpolator, retrieve_track_tuples
from utils.utils import *

SUPPORTED_REPRESENTATIONS = [
    "time_surfaces_v2_5",
    "voxel_grids_5",
    "event_stacks_5",
    "event_stacks_normed_5",
]
MAX_ROTATION_ANGLE = 15
MAX_SCALE_CHANGE_PERCENTAGE = 20
MAX_PERSPECTIVE_THETA = 0.01
MAX_TRANSLATION = 3

torch.multiprocessing.set_sharing_strategy("file_system")


class InputModality(Enum):
    frame = 0
    event = 1


# Data Classes for Baseline Training
@dataclass
class TrackDataConfig:
    frame_paths: list
    event_paths: list
    patch_size: int
    representation: str
    track_name: str
    augment: bool


def recurrent_collate(batch_dataloaders):
    return batch_dataloaders


class TrackData:
    """
    Dataloader for a single feature track. Returns input patches and displacement labels relative to
    the current feature location. Current feature location is either updated manually via accumulate_y_hat()
    or automatically via the ground-truth displacement.
    """

    def __init__(self, track_tuple, config):
        """
        Dataset for a single feature track
        :param track_tuple: (Path to track.gt.txt, track_id)
        :param config:
        """
        self.config = config

        # Track augmentation (disabled atm)
        if False:
            # if config.augment:
            self.flipped_lr = random.choice([True, False])
            self.flipped_ud = random.choice([True, False])
            # self.rotation_angle = round(random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE))
            self.rotation_angle = 0
        else:
            self.flipped_lr, self.flipped_ud, self.rotation_angle = False, False, 0
        self.last_aug_angle, self.last_aug_scale = 0.0, 1.0

        # Get input paths
        self.frame_paths = config.frame_paths
        self.event_paths = config.event_paths

        # TODO: Do this in a non-hacky way
        if "0.0100" in self.event_paths[0]:
            self.index_multiplier = 1
        elif "0.0200" in self.event_paths[0]:
            self.index_multiplier = 2
        else:
            print("Unsupported dt for feature track")
            raise NotImplementedError

        # Input and Labels
        ref_input = read_input(self.frame_paths[0], "grayscale")
        ref_input = augment_input(
            ref_input, self.flipped_lr, self.flipped_ud, self.rotation_angle
        )

        self.track_path, self.track_idx = track_tuple
        self.track_data = np.genfromtxt(self.track_path)
        self.track_data = self.track_data[self.track_data[:, 0] == self.track_idx, 2:]
        self.track_data = augment_track(
            self.track_data,
            self.flipped_lr,
            self.flipped_ud,
            self.rotation_angle,
            (ref_input.shape[1], ref_input.shape[0]),
        )

        self.u_center = self.track_data[0, :]
        self.u_center_gt = self.track_data[0, :]
        self.u_center_init = self.track_data[0, :]

        self.x_ref = get_patch_voxel(ref_input, self.u_center, config.patch_size)

        # Pathing for input data
        self.seq_name = Path(self.track_path).parents[1].stem

        # Operational
        self.time_idx = 0
        self.auto_update_center = False

        # Representation-specific Settings
        if "grayscale" in config.representation:
            self.channels_in_per_patch = 1
        else:
            self.channels_in_per_patch = int(config.representation[-1])
            # in v2, we have separate temporal bins for each event polarity
            if "v2" in config.representation:
                self.channels_in_per_patch *= 2

    def reset(self):
        self.time_idx = 0
        self.u_center = self.u_center_init

    def accumulate_y_hat(self, y_hat):
        """
        Accumulate predicted flows if using predictions instead of gt patches
        :param y_hat: 2-element Tensor
        """
        # Disregard confidence
        y_hat = y_hat[:2]

        # Unaugment the predicted label
        if self.config.augment:
            # y_hat = unaugment_perspective(y_hat.detach().cpu(), self.last_aug_perspective[0], self.last_aug_perspective[1])
            y_hat = unaugment_rotation(y_hat.detach().cpu(), self.last_aug_angle)
            y_hat = unaugment_scale(y_hat, self.last_aug_scale)

            # Translation augmentation
            y_hat += (2 * torch.rand_like(y_hat) - 1) * MAX_TRANSLATION

        self.u_center += y_hat.detach().cpu().numpy().reshape((2,))

    def get_next(self):
        # Increment time
        self.time_idx += 1

        # Round feature location to accommodate get_patch_voxel
        self.u_center = np.rint(self.u_center)

        # Update gt location
        self.u_center_gt = self.track_data[self.time_idx * self.index_multiplier, :]

        # Update total flow
        y = (self.u_center_gt - self.u_center).astype(np.float32)
        y = torch.from_numpy(y)

        # # Update xref (Uncomment if combining frames with events)
        # if self.time_idx % 5 == 0:
        #     frame_idx = self.time_idx // 5
        #     ref_input = read_input(self.frame_paths[frame_idx], 'grayscale')
        #     self.x_ref = get_patch_voxel2(ref_input, self.u_center, self.config.patch_size)

        # Get patch inputs for event representation
        input_1 = read_input(
            self.event_paths[self.time_idx], self.config.representation
        )
        input_1 = augment_input(
            input_1, self.flipped_lr, self.flipped_ud, self.rotation_angle
        )
        x = get_patch_voxel(input_1, self.u_center, self.config.patch_size)
        x = torch.cat([x, self.x_ref], dim=0)

        # Augmentation
        if self.config.augment:
            # Sample rotation and scale
            (
                x[0 : self.channels_in_per_patch, :, :],
                y,
                self.last_aug_scaling,
            ) = augment_scale(
                x[0 : self.channels_in_per_patch, :, :],
                y,
                max_scale_percentage=MAX_SCALE_CHANGE_PERCENTAGE,
            )
            (
                x[0 : self.channels_in_per_patch, :, :],
                y,
                self.last_aug_angle,
            ) = augment_rotation(
                x[0 : self.channels_in_per_patch, :, :],
                y,
                max_rotation_deg=MAX_ROTATION_ANGLE,
            )
            # x[0:self.channels_in_per_patch, :, :], y, self.last_aug_perspective = augment_perspective(x[0:self.channels_in_per_patch, :, :], y,
            #                                                                                           theta=MAX_PERSPECTIVE_THETA)

        # Update center location for next patch
        if self.auto_update_center:
            self.u_center = self.u_center + y.numpy().reshape((2,))

        # Minor Processing Steps
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)

        return x, y


class TrackDataset(Dataset):
    """
    Dataloader for a collection of feature tracks. __getitem__ returns an instance of TrackData.
    """

    def __init__(
        self,
        track_tuples,
        get_frame_paths_fn,
        get_event_paths_fn,
        augment=False,
        patch_size=31,
        track_name="shitomasi_custom",
        representation="time_surfaces_v2_5",
    ):
        super(TrackDataset, self).__init__()
        self.track_tuples = track_tuples
        self.get_frame_paths_fn = get_frame_paths_fn
        self.get_event_paths_fn = get_event_paths_fn
        self.patch_size = patch_size
        self.track_name = track_name
        self.representation = representation
        self.augment = augment
        print(f"Initialized recurrent dataset with {len(self.track_tuples)} tracks.")

    def __len__(self):
        return len(self.track_tuples)

    def __getitem__(self, idx_track):
        track_tuple = self.track_tuples[idx_track]
        data_config = TrackDataConfig(
            self.get_frame_paths_fn(track_tuple[0]),
            self.get_event_paths_fn(track_tuple[0], self.representation),
            self.patch_size,
            self.representation,
            self.track_name,
            self.augment,
        )
        return TrackData(track_tuple, data_config)


class MFDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        extra_dir,
        dt=0.0100,
        batch_size=16,
        num_workers=4,
        patch_size=31,
        augment=False,
        n_train=20000,
        n_val=2000,
        track_name="shitomasi_custom",
        representation="time_surfaces_v2_1",
        mixed_dt=False,
        **kwargs,
    ):
        super(MFDataModule, self).__init__()

        random.seed(1234)

        self.num_workers = num_workers
        self.n_train = n_train
        self.n_val = n_val
        self._has_prepared_data = True

        self.data_dir = Path(data_dir)
        self.extra_dir = Path(extra_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.mixed_dt = mixed_dt
        self.dt = dt
        self.representation = representation
        self.patch_size = patch_size
        self.track_name = track_name

        self.dataset_train, self.dataset_val = None, None

        self.split_track_tuples = {}
        self.split_max_samples = {"train": n_train, "test": n_val}
        for split_name in ["train", "test"]:
            cache_path = (
                self.extra_dir / split_name / ".cache" / f"{track_name}.paths.pkl"
            )
            if cache_path.exists():
                with open(str(cache_path), "rb") as cache_f:
                    track_tuples = pickle.load(cache_f)
            else:
                track_tuples = retrieve_track_tuples(
                    self.extra_dir / split_name, track_name
                )
                with open(str(cache_path), "wb") as cache_f:
                    pickle.dump(track_tuples, cache_f)

            # Shuffle and trim
            n_tracks = len(track_tuples)
            track_tuples_array = np.asarray(track_tuples)
            track_tuples_array = track_tuples_array[: (n_tracks // 64) * 64, :]
            track_tuples_array = track_tuples_array.reshape([(n_tracks // 64), 64, 2])
            rand_perm = np.random.permutation((n_tracks // 64))
            track_tuples_array = track_tuples_array[rand_perm, :, :].reshape(
                (n_tracks // 64) * 64, 2
            )
            track_tuples_array[:, 1] = track_tuples_array[:, 1].astype(np.int)
            track_tuples = []
            for i in range(track_tuples_array.shape[0]):
                track_tuples.append(
                    [track_tuples_array[i, 0], int(track_tuples_array[i, 1])]
                )

            if self.split_max_samples[split_name] < len(track_tuples):
                track_tuples = track_tuples[: self.split_max_samples[split_name]]
            self.split_track_tuples[split_name] = track_tuples

    @staticmethod
    def get_frame_paths(track_path):
        images_dir = Path(
            os.path.split(track_path)[0]
            .replace("_extra", "")
            .replace("tracks", "images")
        )
        return sorted(
            [
                frame_p
                for frame_p in glob(str(images_dir / "*.png"))
                if 400000
                <= int(os.path.split(frame_p)[1].replace(".png", ""))
                <= 900000
            ]
        )

    @staticmethod
    def get_event_paths(track_path, rep, dt):
        event_files = sorted(
            glob(
                str(
                    Path(os.path.split(track_path)[0].replace("tracks", "events"))
                    / f"{random.choice([0.0100, 0.0200]):.4f}"
                    / rep
                    / "*.h5"
                )
            )
        )
        return [
            event_p
            for event_p in event_files
            if 400000 <= int(os.path.split(event_p)[1].replace(".h5", "")) <= 900000
        ]

    @staticmethod
    def get_event_paths_mixed_dt(track_path, rep, dt):
        event_files = sorted(
            glob(
                str(
                    Path(os.path.split(track_path)[0].replace("tracks", "events"))
                    / f"{dt:.4f}"
                    / rep
                    / "*.h5"
                )
            )
        )
        return [
            event_p
            for event_p in event_files
            if 400000 <= int(os.path.split(event_p)[1].replace(".h5", "")) <= 900000
        ]

    def setup(self, stage=None):
        # Create train and val splits
        self.dataset_train = TrackDataset(
            self.split_track_tuples["train"],
            MFDataModule.get_frame_paths,
            partial(MFDataModule.get_event_paths_mixed_dt, dt=self.dt)
            if self.mixed_dt
            else partial(MFDataModule.get_event_paths, dt=self.dt),
            patch_size=self.patch_size,
            track_name=self.track_name,
            representation=self.representation,
            augment=self.augment,
        )
        self.dataset_val = TrackDataset(
            self.split_track_tuples["test"],
            MFDataModule.get_frame_paths,
            partial(MFDataModule.get_event_paths_mixed_dt, dt=self.dt)
            if self.mixed_dt
            else partial(MFDataModule.get_event_paths, dt=self.dt),
            patch_size=self.patch_size,
            track_name="shitomasi_custom",
            representation=self.representation,
            augment=False,
        )

    def train_dataloader(self):
        subseq_sampler = SubSequenceRandomSampler(
            list(range(self.dataset_train.__len__()))
        )

        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=recurrent_collate,
            pin_memory=True,
            sampler=subseq_sampler,
        )

    def val_dataloader(self):
        subseq_sampler = SubSequenceRandomSampler(
            list(range(self.dataset_val.__len__()))
        )

        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=recurrent_collate,
            pin_memory=True,
            sampler=subseq_sampler,
        )


@dataclass
class CornerConfig:
    maxCorners: int
    qualityLevel: float
    minDistance: int
    k: float
    useHarrisDetector: bool
    blockSize: int


# Data Classes for Inference
class EvalDatasetType(Enum):
    EC = 0
    EDS = 1


class SequenceDataset(ABC):
    """
    Data class without ground-truth labels
    """

    def __init__(self):
        self.u_centers, self.u_centers_init = None, None
        self.n_tracks = None
        self.event_first, self.frame_first = None, None
        self.t_now, self.t_init = None, None
        self.n_events, self.n_frames = None, None
        self.patch_size = None
        self.has_poses = False
        self.device = "cpu"
        self.x_ref = torch.zeros(1)

    def initialize(self, max_keypoints=30):
        self.initialize_keypoints(max_keypoints)
        self.initialize_reference_patches()

    def override_keypoints(self, keypoints):
        self.u_centers = keypoints
        self.u_centers = torch.from_numpy(self.u_centers.astype(np.float32))
        self.u_centers_init = self.u_centers.clone()
        self.n_tracks = self.u_centers.shape[0]

        if self.n_tracks == 0:
            raise ValueError("There are no corners in the initial frame")

        self.initialize_reference_patches()

    def initialize_keypoints(self, max_keypoints):
        self.u_centers = cv2.goodFeaturesToTrack(
            self.frame_first,
            max_keypoints,
            qualityLevel=self.corner_config.qualityLevel,
            minDistance=self.corner_config.minDistance,
            k=self.corner_config.k,
            useHarrisDetector=self.corner_config.useHarrisDetector,
            blockSize=self.corner_config.blockSize,
        ).reshape((-1, 2))
        self.u_centers = torch.from_numpy(self.u_centers.astype(np.float32))
        self.u_centers_init = self.u_centers.clone()
        self.n_tracks = self.u_centers.shape[0]

        if self.n_tracks == 0:
            raise ValueError("There are no corners in the initial frame")

    def move_centers(self):
        self.u_centers = self.u_centers.to(self.device)
        self.u_centers_init = self.u_centers_init.to(self.device)
        self.x_ref = self.x_ref.to(self.device)

    def accumulate_y_hat(self, y_hat):
        if y_hat.device != self.device:
            self.device = y_hat.device
            self.move_centers()

        self.u_centers += y_hat.detach()

    def frames(self):
        """
        :return: generator over frames
        """
        pass

    def events(self):
        """
        :return: generator over event representations
        """
        pass



    def get_patches(self, f):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param f:
        :return:
        """
        if f.device != self.device:
            self.device = f.device
            self.move_centers()

        # 0.5 offset is needed due to coordinate system of grid_sample
        return extract_glimpse(
            f.repeat(self.u_centers.size(0), 1, 1, 1),
            (self.patch_size, self.patch_size),
            self.u_centers.detach() + 0.5,
            mode="nearest",
        )

    def get_patches_new(self, arr_h5, padding=4):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param arr_h5: h5 file for the input event representation
        :return: (n_tracks, c, p, p) tensor
        """
        # Extract expanded patches from the h5 files
        u_centers_np = self.u_centers.detach().cpu().numpy()
        x_patches = []
        for i in range(self.n_tracks):
            u_center = u_centers_np[i, :]
            u_center_rounded = np.rint(u_center)
            u_center_offset = (
                u_center - u_center_rounded + ((self.patch_size + padding) // 2.0)
            )
            x_patch_expanded = get_patch_voxel(
                arr_h5, u_center_rounded.reshape((-1,)), self.patch_size + padding
            ).unsqueeze(0)
            x_patch = extract_glimpse(
                x_patch_expanded,
                (self.patch_size, self.patch_size),
                torch.from_numpy(u_center_offset).view((1, 2)) + 0.5,
                mode="nearest",
            )
            x_patches.append(x_patch)
        return torch.cat(x_patches, dim=0)

    @abstractmethod
    def initialize_reference_patches(self):
        pass

    def get_next(self):
        """
        Abstract method for getting input patches and epipolar lines
        :return: input patches (n_corners, C, patch_size, patch_size) and epipolar lines (n_corners, 3)
        """
        pass

    def get_frame(self, image_idx):
        pass





class EDSSubseq(SequenceDataset):
    # ToDo: Add to config file
    pose_r = 3
    pose_mode = False

    def __init__(
        self,
        root_dir,
        sequence_name,
        n_frames,
        patch_size,
        representation,
        dt,
        corner_config,
        include_prev=False,
        fused=False,
        grayscale_ref=True,
        use_colmap_poses=True,
        global_mode=False,
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.patch_size = patch_size
        self.representation = representation
        self.include_prev = include_prev
        # self.dt, self.dt_us = dt, dt * 1e6
        self.grayscale_ref = grayscale_ref
        self.use_colmap_poses = use_colmap_poses
        self.global_mode = global_mode
        self.sequence_dir = self.root_dir / self.sequence_name
        self.corner_config = corner_config

        # Determine number of frames
        self.frame_dir = self.root_dir / sequence_name / "images_corrected"
        max_frames = len(list(self.frame_dir.iterdir())) - 1
        if n_frames == -1 or n_frames > max_frames:
            self.n_frames = max_frames
        else:
            self.n_frames = n_frames

        # Check that event representations have been generated for this dt
        self.dir_representation = (
            self.root_dir
            / sequence_name
            / "event_EST"
            / f"EST_21"
        )

        print('self.dir_representation: ', self.dir_representation)
        if not self.dir_representation.exists():
            print(
                f"{self.representation} has not yet been generated"
            )
            exit()

        # Read timestamps
        self.frame_ts_arr = np.genfromtxt(
            str(self.sequence_dir / "images_timestamps.txt")
        )
        print('self.frame_ts_arr.shape: ', self.frame_ts_arr.shape)

        # Read poses and camera matrix
        if self.use_colmap_poses:
            pose_data_path = self.sequence_dir / "colmap" / "stamped_groundtruth.txt"
        else:
            pose_data_path = self.sequence_dir / "stamped_groundtruth.txt"
        self.pose_data = np.genfromtxt(str(pose_data_path), skip_header=1)
        print('self.pose_data.shape: ', self.pose_data.shape)

        with open(str(self.root_dir / "calib.yaml"), "r") as fh:
            intrinsics = yaml.load(fh, Loader=yaml.SafeLoader)["cam0"]["intrinsics"]
            self.camera_matrix = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ]
            )
            self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        # Tensor Manipulation
        self.channels_in_per_patch = int(self.representation[-1])
        if "v2" in self.representation:
            self.channels_in_per_patch *= 2

        if self.include_prev:   # default: False
            self.cropping_fn = get_patch_voxel_pairs
        else:
            self.cropping_fn = get_patch_voxel

        # Timing and Indices
        self.current_idx = 0
        self.t_init = self.frame_ts_arr[0] * 1e-6
        self.t_end = self.frame_ts_arr[-1] * 1e-6
        self.t_now = self.t_init

        # Pose interpolator for epipolar geometry
        self.pose_interpolator = PoseInterpolator(self.pose_data)
        self.T_last_W = self.pose_interpolator.interpolate(self.t_now)

        # Get counts
        self.n_events = len(glob.glob(f"{self.dir_representation}/*.h5")) # int(np.ceil((self.t_end - self.t_init) / self.dt))

        # Get first imgs
        self.frame_first = cv2.imread(
            str(self.frame_dir / ("frame_" + f"{0}".zfill(10) + ".png")),
            cv2.IMREAD_GRAYSCALE,
        )
        # self.event_first = array_to_tensor(read_input(str(self.dir_representation / '0000000.h5'), self.representation))
        self.resolution = (self.frame_first.shape[1], self.frame_first.shape[0])

        # Extract keypoints, store reference patches
        self.initialize()

    def __len__(self):
        return

    def reset(self):
        self.t_now = self.t_init
        self.current_idx = 0
        self.u_centers = self.u_centers_init

    def initialize_reference_patches(self):
        # Store reference patches
        if "grayscale" in self.representation or self.grayscale_ref:
            ref_input = (
                torch.from_numpy(self.frame_first.astype(np.float32) / 255)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            ref_input = self.event_first.unsqueeze(0)
        self.x_ref = self.get_patches(ref_input)


    def globals(self):
        for i in range(1, self.n_events):
            self.t_now += self.dt
            x = array_to_tensor(
                read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            )

            yield self.t_now, x.unsqueeze(0)

    def events(self):
        self.current_idx = 0    # 1

        for self.current_idx in range(self.current_idx, self.n_events):
            # Get patch inputs and set current time

            self.t_now = self.frame_ts_arr[self.current_idx+1] / 1e6    # += self.dt
            event_input = read_input(
                self.dir_representation
                / f"{str(int(self.current_idx)).zfill(7)}.h5",
                self.representation,
            )


            event_input = np.array(event_input)
            event_input = np.transpose(event_input, (2, 0, 1))
            event_input = torch.from_numpy(event_input).unsqueeze(0).to(self.u_centers.device)


            frame_input = cv2.imread(
                str(
                    self.frame_dir
                    / (
                        "frame_"
                        + f"{self.current_idx+1 // self.pose_r}".zfill(10)
                        + ".png"
                    )
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            frame_input = torch.from_numpy(frame_input).to(self.u_centers.device).unsqueeze(0).unsqueeze(0)


            yield self.t_now, frame_input, event_input


    def frames(self):
        for i in range(1, self.n_frames):
            # Update time info
            self.t_now = self.frame_ts_arr[i] * 1e-6

            frame = cv2.imread(
                str(
                    self.sequence_dir
                    / "images_corrected"
                    / ("frame_" + f"{i}".zfill(10) + ".png")
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            yield self.t_now, frame

    def get_next(self):
        """Strictly for pose supervision"""

        # Update time info
        self.t_now += self.dt

        self.current_idx += 1
        # DEBUG: Use grayscale frame timestamps
        # self.t_now = self.frame_ts_arr[self.current_idx]*1e-6

        # Get patch inputs
        input_1 = read_input(
            self.dir_representation
            / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
            self.representation,
        )
        x = array_to_tensor(input_1)
        x_patches = self.get_patches(x)

        # Get epipolar lines
        T_now_W = self.pose_interpolator.interpolate(self.t_now)
        T_now_last = T_now_W @ np.linalg.inv(self.T_last_W)
        T_last_now = np.linalg.inv(T_now_last)
        self.T_last_W = T_now_W
        F = (
            self.camera_matrix_inv.T
            @ skew(T_last_now[:3, 3])
            @ T_last_now[:3, :3]
            @ self.camera_matrix_inv
        )
        u_centers = self.u_centers.detach().cpu().numpy()
        u_centers_homo = np.concatenate(
            [u_centers, np.ones((u_centers.shape[0], 1))], axis=1
        )
        l_epi = torch.from_numpy(u_centers_homo @ F)

        return x_patches, l_epi


class ECSubseq(SequenceDataset):

    def __init__(
        self,
        root_dir,
        sequence_name,
        n_frames,
        patch_size,
        representation,
        dt,
        corner_config,
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.patch_size = patch_size
        self.representation = representation

        self.sequence_dir = self.root_dir / self.sequence_name
        self.corner_config = corner_config

        # Determine number of frames
        self.pre_frame_dir = self.sequence_dir / "pre_images_corrected"
        self.pre_n_frames = len(list(self.pre_frame_dir.iterdir()))


        self.frame_dir = self.sequence_dir / "images_corrected"
        max_frames = len(list(self.frame_dir.iterdir()))
        if n_frames == -1 or n_frames > max_frames:
            self.n_frames = max_frames
        else:
            self.n_frames = n_frames


        representation_n_bins = 10
        self.warping_event_num = len(glob.glob(os.path.join(self.root_dir, sequence_name, 'pre_events_warping', 'EST_*')))
        print('self.warping_event_num: ', self.warping_event_num)

        ##
        self.pre_warping_event_representation_dir_list = []
        for ttt in range(self.warping_event_num):
            self.pre_warping_event_representation_dir_list.append((
                self.root_dir
                / sequence_name
                / "pre_events_warping"
                / 'EST_{}_{:02d}'.format(representation_n_bins, ttt)
            ))

            if not self.pre_warping_event_representation_dir_list[ttt].exists():
                print(
                    f"warping EST has not yet been generated"
                )
                exit()

        self.pre_exposure_event_representation_dir = (
                self.root_dir
                / sequence_name
                / "pre_events_exposure"
                / 'EST_{}'.format(representation_n_bins)
        )
        if not self.pre_exposure_event_representation_dir.exists():
            print(
                f"exposure EST has not yet been generated"
            )
            exit()


        ##
        self.warping_event_representation_dir_list = []
        for ttt in range(self.warping_event_num):
            self.warping_event_representation_dir_list.append((
                self.root_dir
                / sequence_name
                / "events_warping"
                / 'EST_{}_{:02d}'.format(representation_n_bins, ttt)
            ))

            if not self.warping_event_representation_dir_list[ttt].exists():
                print(
                    f"warping EST has not yet been generated"
                )
                exit()

        self.exposure_event_representation_dir = (
                self.root_dir
                / sequence_name
                / "events_exposure"
                / 'EST_{}'.format(representation_n_bins)
            )
        if not self.exposure_event_representation_dir.exists():
            print(
                f"exposure EST has not yet been generated"
            )
            exit()


        # Read timestamps
        self.pre_frame_ts_arr = np.genfromtxt(str(self.sequence_dir / "pre_images.txt"))
        self.frame_ts_arr = np.genfromtxt(str(self.sequence_dir / "images.txt"))

        # Read poses and camera matrix
        if (self.sequence_dir / "colmap").exists():
            pose_data_path = self.sequence_dir / "colmap" / "stamped_groundtruth.txt"
            self.pose_data = np.genfromtxt(str(pose_data_path), skip_header=1)
        else:
            self.pose_data = np.genfromtxt(str(self.sequence_dir / "groundtruth.txt"))
        intrinsics = np.genfromtxt(str(self.sequence_dir / "calib.txt"))
        self.camera_matrix = np.array(
            [
                [intrinsics[0], 0, intrinsics[2]],
                [0, intrinsics[1], intrinsics[3]],
                [0, 0, 1],
            ]
        )

        # Timing and Indices
        assert self.pre_frame_ts_arr[-1] == self.frame_ts_arr[0], 'The images of prepared data and official data should be continuous'
        self.t_init = self.frame_ts_arr[0]
        self.t_end = self.frame_ts_arr[-1]
        self.t_now = self.t_init

        ## Get counts
        for ttt in range(self.warping_event_num):
            self.pre_n_events = len(glob.glob(f"{self.pre_warping_event_representation_dir_list[ttt]}/*.h5"))
            assert self.pre_n_events == self.pre_n_frames - 1, 'self.pre_n_events: {}, self.pre_n_frames: {}'.format(self.pre_n_events, self.pre_n_frames)

        for ttt in range(self.warping_event_num):
            self.n_events = len(glob.glob(f"{self.warping_event_representation_dir_list[ttt]}/*.h5")) # int(np.ceil((self.t_end - self.t_init) / self.dt))
            assert self.n_events == self.n_frames - 1, 'self.n_events: {}, self.n_frames: {}'.format(self.n_events, self.n_frames)


        ##
        self.frame_first = cv2.imread(
            str(self.frame_dir / ("frame_" + f"{0}".zfill(8) + ".png")),
            cv2.IMREAD_GRAYSCALE,
        )

        pre_frame_end = cv2.imread(
            str(self.pre_frame_dir / ("frame_" + f"{self.pre_n_frames-1}".zfill(8) + ".png")),
            cv2.IMREAD_GRAYSCALE,
        )

        assert np.all(pre_frame_end == self.frame_first), 'The images of prepared data and official data should be continuous'


        self.resolution = (self.frame_first.shape[1], self.frame_first.shape[0])

        # Extract keypoints, store reference patches
        self.initialize()

        print('EC initialization successful!')


    def __len__(self):
        return

    def reset(self):
        self.t_now = self.t_init
        self.u_centers = self.u_centers_init

    def initialize_reference_patches(self):
        # Store reference patches
        ref_input = (
            torch.from_numpy(self.frame_first.astype(np.float32) / 255)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.x_ref = self.get_patches(ref_input)



    def pre_events(self):
        i_start = 0  # 1

        for i in range(i_start, self.pre_n_events):
            # Get patch inputs and set current time
            t_last = self.pre_frame_ts_arr[i]
            self.t_now = self.pre_frame_ts_arr[i + 1]

            ##
            frame_begin = cv2.imread(
                str(
                    self.pre_frame_dir
                    / ("frame_" + f"{i}".zfill(8) + ".png")
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            frame_begin = torch.from_numpy(frame_begin).to(self.u_centers.device).unsqueeze(0).unsqueeze(0).float()


            frame_end = cv2.imread(
                str(
                    self.pre_frame_dir
                    / ("frame_" + f"{i + 1}".zfill(8) + ".png")
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            frame_end = torch.from_numpy(frame_end).to(self.u_centers.device).unsqueeze(0).unsqueeze(0).float()

            ##
            exposure_event_input = read_input(
                self.pre_exposure_event_representation_dir / f"{str(int(i)).zfill(7)}.h5",
                representation='EST',
            )
            exposure_event_input = np.array(exposure_event_input)
            exposure_event_input = np.transpose(exposure_event_input, (2, 0, 1))
            exposure_event_input = torch.from_numpy(exposure_event_input).unsqueeze(0).to(self.u_centers.device).float()

            ##
            warping_event_list = []
            for ttt in range(self.warping_event_num):
                warping_event_input = read_input(
                    self.pre_warping_event_representation_dir_list[ttt] / f"{str(int(i)).zfill(7)}.h5",
                    representation='EST',
                )
                warping_event_input = np.array(warping_event_input)
                warping_event_input = np.transpose(warping_event_input, (2, 0, 1))
                warping_event_list.append(
                    torch.from_numpy(warping_event_input).unsqueeze(0).to(self.u_centers.device).float())


            yield t_last, self.t_now, frame_begin, frame_end, exposure_event_input, warping_event_list,


    def events(self):
        i_start = 0     # 1

        for i in range(i_start, self.n_events):
            # Get patch inputs and set current time
            t_last = self.frame_ts_arr[i]
            self.t_now = self.frame_ts_arr[i+1]

            ##
            frame_begin = cv2.imread(
                str(
                    self.frame_dir
                    / ("frame_" + f"{i}".zfill(8) + ".png")
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            frame_begin = torch.from_numpy(frame_begin).to(self.u_centers.device).unsqueeze(0).unsqueeze(0).float()

            print(str(
                    self.frame_dir
                    / ("frame_" + f"{i}".zfill(8) + ".png")
                ))


            frame_end = cv2.imread(
                str(
                    self.frame_dir
                    / ("frame_" + f"{i + 1}".zfill(8) + ".png")
                ),
                cv2.IMREAD_GRAYSCALE,
            )
            frame_end = torch.from_numpy(frame_end).to(self.u_centers.device).unsqueeze(0).unsqueeze(0).float()

            ##
            exposure_event_input = read_input(
                    self.exposure_event_representation_dir / f"{str(int(i)).zfill(7)}.h5",
                    representation='EST',
                )
            exposure_event_input = np.array(exposure_event_input)
            exposure_event_input = np.transpose(exposure_event_input, (2, 0, 1))
            exposure_event_input = torch.from_numpy(exposure_event_input).unsqueeze(0).to(self.u_centers.device).float()

            ##
            warping_event_list = []
            for ttt in range(self.warping_event_num):
                warping_event_input = read_input(
                    self.warping_event_representation_dir_list[ttt] / f"{str(int(i)).zfill(7)}.h5",
                    representation='EST',
                )
                warping_event_input = np.array(warping_event_input)
                warping_event_input = np.transpose(warping_event_input, (2, 0, 1))
                warping_event_list.append(torch.from_numpy(warping_event_input).unsqueeze(0).to(self.u_centers.device).float())


            yield t_last, self.t_now, frame_begin, frame_end, exposure_event_input, warping_event_list,


    # def frames(self):
    #     # for i in range(self.n_frames):
    #     #     # Update time info
    #     #     self.t_now = self.frame_ts_arr[i]
    #     #
    #     #     frame = cv2.imread(
    #     #         str(
    #     #             self.sequence_dir
    #     #             / "images_corrected"
    #     #             / ("frame_" + f"{i}".zfill(8) + ".png")
    #     #         ),
    #     #         cv2.IMREAD_GRAYSCALE,
    #     #     )
    #     #     yield self.t_now, frame
    #     for i in range(self.n_events):
    #         # Update time info
    #         t_last = self.frame_ts_arr[i]
    #         self.t_now = self.frame_ts_arr[i+1]
    #
    #         frame_begin = cv2.imread(
    #             str(
    #                 self.sequence_dir
    #                 / "images_corrected"
    #                 / ("frame_" + f"{i}".zfill(8) + ".png")
    #             ),
    #             cv2.IMREAD_GRAYSCALE,
    #         )
    #         frame_end = cv2.imread(
    #             str(
    #                 self.sequence_dir
    #                 / "images_corrected"
    #                 / ("frame_" + f"{i + 1}".zfill(8) + ".png")
    #             ),
    #             cv2.IMREAD_GRAYSCALE,
    #         )
    #         yield t_last, self.t_now, frame_begin, frame_end

    def get_next(self):
        raise NotImplementedError


    ## TODO: test
    def detect_keypoints_at_cur_frame(self, frame, max_keypoints):
        detection_result = cv2.goodFeaturesToTrack(
            frame,
            max_keypoints,
            qualityLevel=self.corner_config.qualityLevel,
            minDistance=self.corner_config.minDistance,
            k=self.corner_config.k,
            useHarrisDetector=self.corner_config.useHarrisDetector,
            blockSize=self.corner_config.blockSize,
        ).reshape((-1, 2))

        return torch.from_numpy(detection_result.astype(np.float32))




class SubSequenceRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        # n_samples_per_seq = 8
        n_samples_per_seq = 32
        shifted_start = torch.randint(n_samples_per_seq, [1])
        shifted_indices = self.indices[shifted_start:] + self.indices[-shifted_start:]

        for i in torch.randperm(math.ceil(len(self.indices) / n_samples_per_seq)):
            i_idx = i * n_samples_per_seq

            for i_yield in range(i_idx, min(i_idx + n_samples_per_seq, self.__len__())):
                yield shifted_indices[i_yield]

    def __len__(self) -> int:
        return len(self.indices)
