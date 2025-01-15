""" Predict tracks for a sequence with a network """
import logging
import os
from pathlib import Path

import torch.nn.functional as F
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from tqdm import tqdm
import copy

from utils.dataset import CornerConfig, ECSubseq, EDSSubseq, EvalDatasetType
from utils.timers import CudaTimer, cuda_timers
from utils.track_utils import (
    TrackObserver,
    get_gt_corners,
)

from buffer_manager import FeatureBuffer, TrajectoryBuffer, TimestampBuffer



# Configure GPU order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Logging
logger = logging.getLogger(__name__)
results_table = PrettyTable()
results_table.field_names = ["Inference Time"]

# Configure datasets
corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

EvalDatasetConfigDict = {
    EvalDatasetType.EC: {"dt": 0.010}
}

EVAL_DATASETS = [
    # ("peanuts_light_160_386", EvalDatasetType.EDS),
    # ("rocket_earth_light_338_438", EvalDatasetType.EDS),
    # ("ziggy_in_the_arena_1350_1650", EvalDatasetType.EDS),
    # ("peanuts_running_2360_2460", EvalDatasetType.EDS),

    ("shapes_translation_8_88", EvalDatasetType.EC),
    ("shapes_rotation_165_245", EvalDatasetType.EC),
    ("shapes_6dof_485_565", EvalDatasetType.EC),
    ("boxes_translation_330_410", EvalDatasetType.EC),
    ("boxes_rotation_198_278", EvalDatasetType.EC),
]


@torch.no_grad()
def evaluate(model, sequence_dataset, dt_track_vis, sequence_name, save_path,
             model_height, model_width, model_heatmap_T):
    tracks_pred = TrackObserver(
        t_init=sequence_dataset.t_init, u_centers_init=sequence_dataset.u_centers
    )

    model.eval()
    model.detection_model.eval()
    model.tracking_model.eval()

    print('model reset')
    model.detection_model.reset(torch.tensor([0], device=model.device))

    pre_event_generator = sequence_dataset.pre_events()
    event_generator = sequence_dataset.events()

    cuda_timer = CudaTimer(model.device, sequence_dataset.sequence_name)

    print('model_height, model_width: ', model_height, model_width)


    window_size = 10
    window_step = 1

    buffer_max_len = window_size * 2
    feature_buffer = FeatureBuffer(buffer_max_len)
    traj_buffer = TrajectoryBuffer(buffer_max_len)
    timestamp_buffer = TimestampBuffer(buffer_max_len)

    official_first_time = True

    query_points = None
    query_vectors = None

    with torch.no_grad():
        ## Preparation
        for pre_idx, (pre_t_last, pre_t, pre_frame_begin, _, pre_exposure_event, pre_warping_event_list) in enumerate(tqdm(
                pre_event_generator,
                total=sequence_dataset.pre_n_events,
                desc="Prepare stage, Predicting tracks with network...",
        )):
            # print('pre_idx: pre_t: '.format(pre_idx, pre_t))
            assert model_heatmap_T == len(pre_warping_event_list)+1


            ##
            pre_warping_event_list = [torch.zeros_like(pre_warping_event_list[0])] + pre_warping_event_list
            # print('len(warping_event_list): ', len(pre_warping_event_list))


            ## resize
            pre_frame_input = pre_frame_begin.to(model.device)

            raw_H, raw_W = pre_frame_input.shape[-2], pre_frame_input.shape[-1]
            pre_frame_input = F.interpolate(pre_frame_input, size=(model_height, model_width),
                                        mode='bilinear')  # [B=1, 1, H, W]
            pre_exposure_event = F.interpolate(pre_exposure_event.to(model.device), size=(model_height, model_width),
                                           mode='bilinear')
            for ttt, warping_event in enumerate(pre_warping_event_list):
                pre_warping_event_list[ttt] = F.interpolate(warping_event.to(model.device),
                                                        size=(model_height, model_width),
                                                        mode='bilinear')  # [1, C, H, W]

            ##
            pred_dict = model.detection_model(frame=pre_frame_input.float() / 255,
                                              exposure_event=pre_exposure_event,
                                              warping_event_list=pre_warping_event_list)
            pred_features = pred_dict['desc_list']
            pred_features = torch.stack(pred_features, dim=0)  # [heatmap_T, B=1, 128, Hc, Wc]
            heatmap_T = pred_features.shape[0]
            # assert heatmap_T == len(pre_warping_event_list) + 1

            ##
            delta_t = pre_t - pre_t_last
            ts_from_multi_time_steps = []  #
            interval_time = delta_t / (heatmap_T - 1)
            for sub_i in range(heatmap_T):
                ts_from_multi_time_steps.append(pre_t_last + sub_i * interval_time)
            ts_from_multi_time_steps = torch.tensor(ts_from_multi_time_steps)  # [heatmap_T]
            # print('ts_from_multi_time_steps: ', ts_from_multi_time_steps)

            ##
            feature_buffer.push(pred_features)
            traj_buffer.push(None)
            timestamp_buffer.push(ts_from_multi_time_steps)




        #
        for idx, (t_last, t, frame_begin, _, exposure_event, warping_event_list) in enumerate(tqdm(
            event_generator,
            total=sequence_dataset.n_events,
            desc="Official Stage, Predicting tracks with network...",
        )):
            # print('idx: t: '.format(idx, t))
            assert model_heatmap_T == len(warping_event_list) + 1


            ##
            warping_event_list = [torch.zeros_like(warping_event_list[0])] + warping_event_list
            # print('len(warping_event_list): ', len(warping_event_list))


            ##
            frame_input = frame_begin.to(model.device)

            raw_H, raw_W = frame_input.shape[-2], frame_input.shape[-1]
            # print('raw_H, raw_W: ', raw_H, raw_W)
            frame_input = F.interpolate(frame_input, size=(model_height, model_width), mode='bilinear')  # [B=1, 1, H, W]
            exposure_event = F.interpolate(exposure_event.to(model.device), size=(model_height, model_width), mode='bilinear')
            for ttt, warping_event in enumerate(warping_event_list):
                warping_event_list[ttt] = F.interpolate(warping_event.to(model.device), size=(model_height, model_width),
                                                        mode='bilinear')  # [1, C, H, W]


            ##
            pred_dict = model.detection_model(frame=frame_input.float() / 255,
                                              exposure_event=exposure_event,
                                              warping_event_list=warping_event_list)
            pred_features = pred_dict['desc_list']
            pred_features = torch.stack(pred_features, dim=0)  # [heatmap_T, B=1, 128, Hc, Wc]
            heatmap_T = pred_features.shape[0]
            # assert heatmap_T == len(warping_event_list) + 1



            ##
            delta_t = t - t_last
            ts_from_multi_time_steps = []  #
            interval_time = delta_t / (heatmap_T - 1)
            for sub_i in range(heatmap_T):
                ts_from_multi_time_steps.append(t_last + sub_i * interval_time)
            ts_from_multi_time_steps = torch.tensor(ts_from_multi_time_steps)  # [heatmap_T]
            # print('ts_from_multi_time_steps: ', ts_from_multi_time_steps)


            assert len(feature_buffer) == len(timestamp_buffer) != 0
            assert len(feature_buffer) == len(timestamp_buffer) >= window_size
            if official_first_time: #
                assert len(traj_buffer) == 0
                assert timestamp_buffer.buffer[-1] == ts_from_multi_time_steps[0]


                ##
                query_points = sequence_dataset.u_centers.unsqueeze(0).permute(0, 2, 1).to(frame_input.device).float()  # [B=1, 2(xy), N]
                query_points[..., 0, :] = query_points[..., 0, :] * (model_width / raw_W)   # resize
                query_points[..., 1, :] = query_points[..., 1, :] * (model_height / raw_H)

                ##
                traj_buffer.buffer = query_points.unsqueeze(0).repeat(len(feature_buffer), 1, 1, 1)  # [T, B=1, 2(xy), N]

                ## extract vector
                query_idx = torch.zeros((1, query_points.shape[-1])).long().to(frame_input.device)  # [B=1, N]
                query_vectors = model.tracking_model.extract_query_vectors(query_idx, query_points, pred_features)  # [B=1, C, N]


                official_first_time = False


            assert query_points is not None
            assert query_vectors is not None
            assert len(feature_buffer) >= window_size - window_step, '{}, {}'.format(len(feature_buffer), window_size - window_step)


            assert heatmap_T % window_step == 0
            step_num = heatmap_T // window_step
            for i in range(step_num):
                # print('\ni: ', i)
                # print('i*window_step, (i+1)*window_step: ', i * window_step, (i + 1) * window_step)

                feature_new = pred_features[i * window_step:(i + 1) * window_step]  # [window_step, B=1, 2, N]
                traj_new = traj_buffer.buffer[-1:].repeat(window_step, 1, 1, 1)  # [window_step, B=1, 2, N]
                ts_new = ts_from_multi_time_steps[i * window_step:(i + 1) * window_step]  # [window_step]

                assert len(feature_buffer) == len(traj_buffer) == len(timestamp_buffer)
                old_window_len = window_size - window_step
                feature_old = feature_buffer.buffer[-old_window_len:]
                traj_old = traj_buffer.buffer[-old_window_len:]
                ts_old = timestamp_buffer.buffer[-old_window_len:]

                feature_for_tracker = torch.cat([feature_old, feature_new], dim=0).to(model.device)
                traj_for_tracker = torch.cat([traj_old, traj_new], dim=0).to(model.device)
                ts_for_tracker = torch.cat([ts_old, ts_new], dim=0).to(model.device)

                refine_traj = model.tracking_model(
                    traj_init=traj_for_tracker,
                    query_vectors_init=query_vectors,
                    features=feature_for_tracker)[-1]  # [T, B, 2, N]

                ##
                feature_buffer.push(feature_new)
                traj_buffer.push(traj_new)
                traj_buffer.buffer[-window_size:] = refine_traj
                timestamp_buffer.push(ts_new)

                assert len(feature_buffer) == len(traj_buffer) == len(timestamp_buffer)


                ##
                points_for_save = copy.deepcopy(traj_buffer.buffer[-1])  # [B=1, 2, N]
                points_for_save[..., 0, :] = points_for_save[..., 0, :] / (model_width / raw_W)
                points_for_save[..., 1, :] = points_for_save[..., 1, :] / (model_height / raw_H)
                points_for_save = points_for_save.squeeze(0).permute(1, 0).cpu().numpy()    # [N, 2(xy)]

                ts_for_save = timestamp_buffer.buffer[-1].float().item()
                tracks_pred.add_observation(ts_for_save, points_for_save)



    ## Save predicted tracks
    np.savetxt(
        os.path.join(save_path, f"{sequence_name}.txt"),
        tracks_pred.track_data,
        fmt=["%i", "%.9f", "%i", "%i"],
        delimiter=" ",
    )

    metrics = {}
    metrics["latency"] = sum(cuda_timers[sequence_dataset.sequence_name])

    return metrics


@hydra.main(config_path="configs", config_name="eval_real_defaults")
def track(cfg):

    model, model_height, model_width, model_heatmap_T = init_model(cfg.project_dir)
    model.eval()


    # Run evaluation on each dataset
    for seq_name, seq_type in EVAL_DATASETS:
        print('\nseq_name: ', seq_name)
        print('seq_type: ', seq_type)

        if seq_type == EvalDatasetType.EC:
            dataset_class = ECSubseq
        elif seq_type == EvalDatasetType.EDS:
            dataset_class = EDSSubseq
        else:
            raise ValueError

        dataset = dataset_class(
            root_dir=cfg.EC_path,       # EvalDatasetConfigDict[seq_type]["root_dir"],
            sequence_name=seq_name,
            n_frames=-1,
            patch_size=cfg.patch_size,
            representation=cfg.representation,
            dt=EvalDatasetConfigDict[seq_type]["dt"],
            corner_config=corner_config,
        )

        # Load ground truth corners for this seq and override initialization
        # gt_features_path = str(Path(cfg.gt_path) / f"{seq_name}.gt.txt")

        gt_path = str(os.path.join(cfg.project_dir, 'gt_tracks'))
        gt_features_path = str(Path(gt_path) / f"{seq_name}.gt.txt")
        gt_start_corners = get_gt_corners(gt_features_path)

        save_path = str(os.path.join(cfg.project_dir, 'eval_results/FF-KDT/'))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        dataset.override_keypoints(gt_start_corners)

        metrics = evaluate(model, dataset, cfg.dt_track_vis, seq_name, save_path,
                           model_height=model_height, model_width=model_width,
                           model_heatmap_T=model_heatmap_T)

        logger.info(f"=== DATASET: {seq_name} ===")
        logger.info(f"Latency: {metrics['latency']} s")

        results_table.add_row([metrics["latency"]])

    logger.info(f"\n{results_table.get_string()}")



import argparse
from model_tracking.A_lightning_model_tracking_net_light_v2 import CornerDetectionLightningModel
def init_model(project_dir):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cpu', default=False, help='use cpu')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(project_dir, 'FF-KDT_checkpoints/epoch=8-step=45000.ckpt'),
                        help='checkpoint')
    parser.add_argument('--data_device', type=str, default='cuda:0', help='run simulation on the cpu/gpu')

    parser.add_argument('--warping_event_volume_depth', type=int, default=10, help='event volume depth')
    parser.add_argument('--exposure_event_volume_depth', type=int, default=10, help='event volume depth')

    parser.add_argument('--needed_number_of_heatmaps', type=int, default=5, help='number of target corner heatmaps, heatmap_T')
    parser.add_argument('--num_tbins', type=int, default=5, help="timesteps per batch tbppt")
    parser.add_argument('--height', type=int, default=240, help='image height in evaluation')
    parser.add_argument('--width', type=int, default=320, help='image width in evaluation')


    params, _ = parser.parse_known_args(None)
    print('pl version: ', pl.__version__)
    params.warping_cin = params.warping_event_volume_depth * 2  # pos and neg
    params.exposure_cin = params.exposure_event_volume_depth * 2  # pos and neg
    params.cout = params.needed_number_of_heatmaps
    print(params)

    model = CornerDetectionLightningModel(params, None, None)
    if not params.cpu:
        model.cuda()
    else:
        params.data_device = "cpu"

    ckpt = params.checkpoint
    # print('ckpt: ', ckpt)

    checkpoint = torch.load(ckpt, map_location=torch.device('cpu') if params.cpu else torch.device("cuda"))
    model.load_state_dict(checkpoint['state_dict'])

    return model, params.height, params.width, params.cout


if __name__ == "__main__":
    track()
