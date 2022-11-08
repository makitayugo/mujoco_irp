from typing import Tuple, Optional
import torch
import numpy as np
import mujoco_py
from scipy.interpolate import interp1d
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
#import gym
from common.sample_util import VirtualSampleGrid, transpose_data_dict
from abr_control.arms.mujoco_config import MujocoConfig
from abr_control.controllers import Joint
from abr_control.interfaces.mujoco import Mujoco
import os
import yaml
from abr_control_mod.mujoco_utils import (
    get_rope_body_ids, get_body_center_of_mass,
    apply_impulse_com_batch, get_mujoco_state, set_mujoco_state)
from common.urscript_control_util import get_movej_trajectory
from common.template_util import require_xml
from common.cv_util import get_traj_occupancy
from common.sample_util import (GridCoordTransformer, get_nd_index_volume)
from typing import Optional, Union
import pathlib
#from environments.goal_selection import select_rope_and_goals
import zarr
from common.sample_util import get_nd_index_volume

from networks.delta_trajectory_deeplab import DeltaTrajectoryDeeplab
from real_ur5.delta_action_sampler import DeltaActionGaussianSampler
from real_ur5.delta_action_selector import DeltaActionSelector

def deg_to_rad(deg):
    return deg / 180 * np.pi


def get_param_dict(length, density,
                   width=0.02, stick_length=0.48,
                   num_nodes=25, **kwargs):
    stiffness = 0.01 / 0.015 * density
    damping = 0.005 / 0.015 * density

    link_size = length / (num_nodes - 1) / 2
    param_dict = {
        'count': num_nodes,
        'spacing': link_size * 2,
        'link_size': link_size,
        'link_width': width / 2,
        'link_mass': density * length / num_nodes,
        'stiffness': stiffness,
        'damping': damping,
        'stick_size': (stick_length - link_size) / 2,
        'ee_offset': stick_length
    }
    return param_dict


class SimEnvironment:
#    def __init__(self, env_cfg: DictConfig, rope_cfg: DictConfig):

    def __init__(self, env_cfg: DictConfig, rope_cfg: DictConfig, render_mode: str):
    #def __init__(self, render_mode: str):
        self.env_cfg = env_cfg
        self.render_mode = render_mode
        self._viewers = {}
        self.viewer = None

        # create transformer
        transformer = GridCoordTransformer(**env_cfg.transformer)

        # build simulation xml
        xml_dir = to_absolute_path(env_cfg.xml_dir)
        rope_param_dict = get_param_dict(**rope_cfg)
        xml_fname = require_xml(
            xml_dir, rope_param_dict,
            to_absolute_path(env_cfg.template_path), force=True)

        # load mujoco environment
        robot_config = MujocoConfig(
            xml_file=xml_fname,
            folder=xml_dir)
        interface = Mujoco(robot_config,
                           dt=env_cfg.sim.dt,
                           visualize=env_cfg.sim.visualize)
        interface.connect()
        ctrlr = Joint(robot_config, kp=env_cfg.sim.kp)

        j_init = deg_to_rad(np.array(env_cfg.sim.j_init_deg))
        interface.set_joint_state(q=j_init, dq=np.zeros(6))
        init_state = get_mujoco_state(interface.sim)
        rope_body_ids = get_rope_body_ids(interface.sim.model)

        # viewer = mujoco_py.MjViewer(rope_body_ids)
        # viewer.render()

        self.transformer = transformer
        self.interface = interface
        self.ctrlr = ctrlr
        self.init_state = init_state
        self.rope_body_ids = rope_body_ids
        self.rs = np.random.RandomState(env_cfg.seed)

    def set_goal(self, goal: Tuple[float, float]):
        self.goal_pix = tuple(self.transformer.to_grid([goal], clip=True)[0])

    def set_goal_pix(self, goal_pix: Tuple[int, int]):
        self.goal_pix = tuple(goal_pix)

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, dict]:
        interface = self.interface
        rope_body_ids = self.rope_body_ids
        ctrlr = self.ctrlr
        init_state = self.init_state

        if self.goal_pix is None:
            raise RuntimeError('Please call set_goal before step.')

        eps = 1e-7
        action = np.clip(action, 0, 1 - eps)
        # compute action
        ac = self.env_cfg.action
        speed_interp = interp1d([0, 1], ac.speed_range)
        j2_interp = interp1d([0, 1], ac.j2_delta_range)
        j3_interp = interp1d([0, 1], ac.j3_delta_range)
        speed = speed_interp(action[0])
        j2_delta = j2_interp(action[1])
        j3_delta = j3_interp(action[2])
        impulse = 0
        if self.env_cfg.random_init:
            impulse = self.rs.uniform(*ac.impulse_range)

        # generate target
        sc = self.env_cfg.sim
        j_init = deg_to_rad(np.array(sc.j_init_deg))
        j_start = j_init
        j_end = j_init.copy()
        j_end[2] += j2_delta
        j_end[3] += j3_delta

        q_target = get_movej_trajectory(
            j_start=j_start, j_end=j_end,
            acceleration=ac.acceleration, speed=speed, dt=sc.dt)
        qdot_target = np.gradient(q_target, sc.dt, axis=0)

        # run simulation
        set_mujoco_state(interface.sim, init_state)

        impulses = np.multiply.outer(
            np.linspace(0, 1, len(rope_body_ids)),
            np.array([impulse, 0, 0]))
        apply_impulse_com_batch(
            sim=interface.sim,
            body_ids=rope_body_ids,
            impulses=impulses)

        num_sim_steps = int(sc.sim_duration / sc.dt)
        rope_history = list()
        n_contact_buffer = [0]
        for i in range(num_sim_steps):
            feedback = interface.get_feedback()

            idx = min(i, len(q_target) - 1)
            u = ctrlr.generate(
                q=feedback['q'],
                dq=feedback['dq'],
                target=q_target[idx],
                target_velocity=qdot_target[idx]
            )
            n_contact = interface.sim.data.ncon
            n_contact_buffer.append(n_contact)
            if i % sc.subsample_rate == 0:
                nc = max(n_contact_buffer)
                if nc > 0:
                    break
                rope_body_com = get_body_center_of_mass(
                    interface.sim.data, rope_body_ids)
                rope_history.append(rope_body_com[-1])
                n_contact_buffer = [0]
            interface.send_forces(u)

        this_data = np.array(rope_history, dtype=np.float32)
        traj_img = get_traj_occupancy(this_data[:, [0, 2]], self.transformer)

        img_coords = get_nd_index_volume(traj_img.shape)
        traj_coords = img_coords[traj_img]
        dists_pix = np.linalg.norm(traj_coords - self.goal_pix, axis=-1)
        dist_pix = np.min(dists_pix)
        dist_m = dist_pix / self.transformer.pix_per_m

        # return
        observation = traj_img
        loss = dist_m
        done = False
        info = {
            'action': action
        }
        return observation, loss, done, info

    def _get_viewer(
        self, mode
    ) -> Union["mujoco_py.MjViewer", "mujoco_py.MjRenderContextOffscreen"]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.interface.sim)

            elif mode in {"rgb_array", "depth_array"}:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.interface.sim, -1)
            else:
                raise AttributeError(
                    f"Unknown mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

#        width, height = self.width, self.height
        width, height = 640, 480
#        width, height = 1280, 800

 #       camera_name, camera_id = self.camera_name, self.camera_id
        camera_name, camera_id = None, 0

        if self.render_mode in {"rgb_array", "depth_array"}:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                if camera_name in self.model._camera_name2id:
                    camera_id = self.model.camera_name2id(camera_name)

                self._get_viewer(self.render_mode).render(
                    width, height, camera_id=camera_id
                )

        if self.render_mode == "rgb_array":
            data = self._get_viewer(self.render_mode).read_pixels(
                width, height, depth=False
            )
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif self.render_mode == "depth_array":
            self._get_viewer(self.render_mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(self.render_mode).read_pixels(
                width, height, depth=True
            )[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif self.render_mode == "human":
            self._get_viewer(self.render_mode).render()

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """

#@hydra.main(config_path="/home/kenkyo2020/irp/config")
#@hydra.main(config_path="config", config_name="sim_environment_3")
def select_rope_and_goals(
        zarr_path, n_ropes, n_goals,
        mask_names=('split/is_test',),
        seed=0):
    root = zarr.open(to_absolute_path(zarr_path), 'r')
    assert(len(mask_names) > 0)
    mask = None
    for name in mask_names:
        this_mask = root[name][:]
        if mask is None:
            mask = this_mask
        else:
            mask = mask & this_mask

    rope_idx_volume = get_nd_index_volume(mask.shape)
    test_rope_idxs = rope_idx_volume[mask]

    rs = np.random.RandomState(seed=seed)
    rope_ids = test_rope_idxs[
        rs.choice(len(test_rope_idxs), 
        size=n_ropes, replace=False)]
    # select goals
    max_hitrate_array = root['control/max_hitrate']
    img_shape = max_hitrate_array.shape[-2:]
    goal_coord_img = get_nd_index_volume(img_shape)
    rope_goal_dict = dict()
    for i in range(len(rope_ids)):
        rope_id = rope_ids[i]
        this_hitrate_img = max_hitrate_array[tuple(rope_id)]
        valid_goal_mask = this_hitrate_img > 0.95
        valid_goals = goal_coord_img[valid_goal_mask]
        rope_goal_dict[tuple(rope_id.tolist())] = valid_goals[
            rs.choice(len(valid_goals), n_goals, replace=False)]
    return rope_goal_dict
@hydra.main(config_path="/home/kenkyo2020/irp/config", config_name=pathlib.Path(__file__).stem)

def main(cfg: DictConfig) -> None:
    if not cfg.offline:
        wandb.init(**cfg.wandb)
    abs_zarr_path = to_absolute_path(cfg.setup.zarr_path)
    rope_goal_dict = select_rope_and_goals(
        zarr_path=abs_zarr_path,
        **cfg.setup.selection)
    config = OmegaConf.to_container(cfg, resolve=True)
    output_dir = os.getcwd()
    config['output_dir'] = output_dir
    yaml.dump(config, open('sim_environment_3.yaml', 'w'), default_flow_style=False)
    if not cfg.offline:
        wandb.config.update(config)

    root = zarr.open(abs_zarr_path, 'r')
    init_action_array = root['train_rope/best_action_coord']
    action_scale = np.array(root[cfg.setup.name].shape[2:5])
    sample_grid = VirtualSampleGrid.from_zarr_group(root)

    # load action model
    device = torch.device('cuda', cfg.action.gpu_id)
    dtype = torch.float16 if cfg.action.use_fp16 else torch.float32
    sampler = DeltaActionGaussianSampler(**cfg.action.sampler)
    action_model = DeltaTrajectoryDeeplab.load_from_checkpoint(
        to_absolute_path(cfg.action.ckpt_path))
#    action_model_gpu = action_model.to(
#        device, dtype=dtype).eval()
#    selector = DeltaActionSelector(
#        model=action_model_gpu, **cfg.action.selector)
    selector = DeltaActionSelector(
        model=action_model, **cfg.action.selector)


    # load action model
    ropes_log = list()
    for rope_id, goals in rope_goal_dict.items():
        rope_config = {
            'length': sample_grid.dim_samples[0][rope_id[0]],
            'density': sample_grid.dim_samples[1][rope_id[1]]
        }
        env = SimEnvironment(
            env_cfg=cfg.env,
            rope_cfg=rope_config,
            render_mode="rgb_array")
            # "human""rgb_array"
        while True:
#            env.step(env.action_space.sample())
            env.render()


if __name__ == "__main__":
    main()