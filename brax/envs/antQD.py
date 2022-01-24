# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains an ant to run in the +x direction."""

import brax
from brax import jumpy as jp
from brax.envs import ant, env


class AntQD(ant.Ant):
    """Trains an ant to run in the +x direction."""
    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe ant body position and velocities."""
        # some pre-processing to pull joint angles and velocities
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

        # qpos:
        # Z of the torso (1,)
        # orientation of the torso as quaternion (4,)
        # joint angles (8,)
        qpos = [qp.pos[0, :], qp.rot[0], joint_angle]

        # qvel:
        # velcotiy of the torso (3,)
        # angular velocity of the torso (3,)
        # joint angle velocities (8,)
        qvel = [qp.vel[0], qp.ang[0], joint_vel]

        # external contact forces:
        # delta velocity (3,), delta ang (3,) * 10 bodies in the system
        # Note that mujoco has 4 extra bodies tucked inside the Torso that Brax
        # ignores
        cfrc = [
            jp.clip(info.contact.vel, -1, 1),
            jp.clip(info.contact.ang, -1, 1)
        ]
        # flatten bottom dimension
        cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]

        return jp.concatenate(qpos + qvel + cfrc)
