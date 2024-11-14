import asyncio
import numpy as np
import pinocchio as pin
from os.path import join, abspath, dirname
import math

class ArmDynamics:
    def __init__(self, urdf_path):
        """Initialize the 2-DOF planar arm using a URDF file."""
        # Load the URDF model
        self.model = pin.buildModelFromUrdf(urdf_path)
        # self.model.gravity.linear = np.array([0, -9.81, 0])
        self.data = self.model.createData()

    def compute_dynamics(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics for gravity compensation using URDF model.
        
        Args:
            q: Joint positions in radians
            v: Joint velocities in radians/s
        
        Returns:
            tau: Joint torques accounting only for gravity
        """
        # Convert rotations to radians
        q_rad = np.array([pos * 2 * np.pi for pos in q])
        v_rad = np.array([vel * 2 * np.pi for vel in v])

        # Zero acceleration for gravity compensation only
        a = np.zeros_like(q_rad)

        # Compute torques using RNEA for gravity compensation
        tau_gravity = pin.rnea(self.model, self.data, q_rad, v_rad, a)
        
        return tau_gravity
    
# Path to the URDF file
urdf_path = join(dirname(abspath(__file__)), "arm.urdf")

# Initialize the arm model
arm = ArmDynamics(urdf_path)

q = [0, 0] # rads
v = [0, 0] # rads/s

tau = arm.compute_dynamics(q, v)
print(tau)