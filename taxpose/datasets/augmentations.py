from dataclasses import dataclass
from typing import Callable, Optional
import re

import numpy as np
import numpy.typing as npt
import torch
from pytorch3d.ops import sample_farthest_points

from taxpose.utils.occlusion_utils import ball_occlusion, plane_occlusion


def maybe_downsample(
    points: npt.NDArray[np.float32], num_points: Optional[int] = None, type: str = "fps"
) -> npt.NDArray[np.float32]:
    if num_points is None:
        return points

    if points.shape[1] < num_points:
        raise ValueError("Cannot downsample to more points than exist in the cloud.")

    if re.match(r"^fps$", type) is not None:
        return sample_farthest_points(torch.from_numpy(points), K=num_points, random_start_point=True)[0].numpy()
    elif re.match(r"^random$", type) is not None:
        random_idx = torch.randperm(points.shape[1])[:num_points]
        return points[:, random_idx]
    elif re.match(r"^random_0\.[0-9]$", type) is not None:
        prob = float(re.match(r"^random_(0\.[0-9])$", type).group(1))
        if np.random.random() > prob:
            return sample_farthest_points(torch.from_numpy(points), K=num_points, random_start_point=True)[0].numpy()
        else:
            random_idx = torch.randperm(points.shape[1])[:num_points]
            return points[:, random_idx]
    elif re.match(r"^[0-9]+N_random_fps$", type) is not None:
        random_num_points = int(re.match(r"^([0-9]+)N_random_fps$", type).group(1)) * num_points
        random_idx = torch.randperm(points.shape[1])[:random_num_points]
        random_points = points[:, random_idx]
        return sample_farthest_points(torch.from_numpy(random_points), K=num_points, random_start_point=True)[0].numpy()
    else:
        return points


@dataclass
class OcclusionConfig:
    occlusion_class: int

    # Ball occlusion.
    action_ball_occlusion: bool = True
    action_ball_radius: float = 0.1
    action_plane_occlusion: bool = True
    action_plane_standoff: float = 0.04
    
    anchor_ball_occlusion: bool = True
    anchor_ball_radius: float = 0.1
    anchor_plane_occlusion: bool = True
    anchor_plane_standoff: float = 0.04

    occlusion_prob: float = 0.5
    
    downsample_type: str = "fps" 

def occlusion_fn(
    cfg: Optional[OcclusionConfig] = None,
) -> Callable[[npt.NDArray[np.float32], int, int], npt.NDArray[np.float32]]:
    if cfg is None:
        return lambda x, y, z: x

    def occlusion(points: npt.NDArray[np.float32], obj_class: int, min_num_points: int, type: str = "action"):
        if getattr(cfg, f"{type}_ball_occlusion"):
            if np.random.rand() < cfg.occlusion_prob:
                points_new, _ = ball_occlusion(points[0], radius=getattr(cfg, f"{type}_ball_radius"))

                # Ignore the occlusion if it's going to mess us up later...
                if points_new.shape[0] > min_num_points:
                    points = points_new.unsqueeze(0)

        if getattr(cfg, f"{type}_plane_occlusion"):
            if np.random.rand() < cfg.occlusion_prob:
                points_new, _ = plane_occlusion(
                    points[0], stand_off=getattr(cfg, f"{type}_plane_standoff")
                )
                # Ignore the occlusion if it's going to mess us up later...
                if points_new.shape[0] > min_num_points:
                    points = points_new.unsqueeze(0)
        return points if isinstance(points, np.ndarray) else points.numpy()

    return occlusion
