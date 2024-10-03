import asyncio
import math
import moteus
import moteus_pi3hat
import time

import numpy as np
import pydrake.all as drake

class Arm:
    def __init__(self, m1, m2, l1, l2, initial_end_mass):
        """Initialize the 2-DOF arm with initial parameters."""
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        
        # Create the multibody plant
        self.plant = drake.multibody.plant.MultibodyPlant(time_step=0.0)
        
        # Create links
        self.link1 = self.plant.AddRigidBody(
            "link1",
            drake.SpatialInertia(
                mass=m1,
                p_PScm_E=np.array([l1/2, 0, 0]),
                G_SP_E=drake.RotationalInertia(m1*l1**2/12, m1*l1**2/12, m1*l1**2/12)
            )
        )
        
        # For link2, we'll use a parameter that can be updated
        self.link2 = self.plant.AddRigidBody("link2", drake.SpatialInertia())
        self.end_mass_param = self.plant.AddParameter(
            drake.multibody.Parameter(1)  # 1-dimensional parameter
        )
        
        # Add joints
        self.shoulder = self.plant.AddRevoluteJoint(
            "shoulder",
            self.plant.world_frame(),
            self.link1.body_frame(),
            [0, 0, 1]
        )
        self.elbow = self.plant.AddRevoluteJoint(
            "elbow",
            self.link1.body_frame(),
            self.link2.body_frame(),
            [0, 0, 1]
        )
        
        # Add gravity
        self.plant.mutable_gravity_field().set_gravity_vector([0, -9.81, 0])
        
        # Finalize the plant
        self.plant.Finalize()
        
        # Create a context for the plant
        self.context = self.plant.CreateDefaultContext()
        
        # Set the initial end mass
        self.set_end_mass(initial_end_mass)

    def set_end_mass(self, end_mass):
        """Update the end mass and recalculate link2 properties."""
        total_mass2 = self.m2 + end_mass
        com2 = (self.m2 * self.l2/2 + end_mass * self.l2) / total_mass2
        I2 = self.m2 * (self.l2/2)**2 + end_mass * self.l2**2  # Simple approximation

        # Update the parameter value
        self.plant.SetParameter(self.context, self.end_mass_param, [end_mass])
        
        # Update link2 spatial inertia
        M2 = drake.SpatialInertia(
            mass=total_mass2,
            p_PScm_E=np.array([com2, 0, 0]),
            G_SP_E=drake.RotationalInertia(I2, I2, I2)
        )
        self.plant.SetBodySpatialInertiaInBodyFrame(self.context, self.link2, M2)

    def calculate_inverse_dynamics(self, q, v, vd):
        """Calculate inverse dynamics using current end mass."""
        # Set the state
        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, v)
        
        # Calculate inverse dynamics
        return self.plant.CalcInverseDynamics(
            self.context, vd, drake.multibody.plant.MultibodyForces(self.plant)
        )

async def main():
    transport = moteus_pi3hat.Pi3HatRouter(
        servo_bus_map = {
        1 : [1],
        }
    )

    servos = {
        servo_id : moteus.Controller(id=servo_id, transport=transport)
        for servo_id in [1]
    }

    results = await transport.cycle([x.make_stop(query=True) for x in servos.values()])


    print("Initial Values")
    print(", ".join(
        f"({result.arbitration_id}) " 
        + f"({result.values[moteus.Register.POSITION]}) " 
        + f"({result.values[moteus.Register.VELOCITY]})"  
        + f"({result.values[moteus.Register.ACCELARATION]})"  
        for result in results)
        )
    
    positions = [result.values[moteus.Register.POSITION] for result in results] * 2 * math.pi
    velocity = [result.values[moteus.Register.VELOCITY] for result in results]
    accel = [result.values[moteus.Register.ACCELARATION] for result in results]

    print("\nStarting loop")
    # while True:
    #     dp, dv, dtau = inverse_kinematic(positions, velocity, accel)

    #     commands = [
    #         servos[1].make_position(
    #         feedforward_torque=dtau,
    #         query=True)
    #     ]

    #     results = await transport.cycle(commands)


    #     print(", ".join(
    #     f"({result.arbitration_id}) " 
    #     + f"({result.values[moteus.Register.POSITION]}) " 
    #     + f"({result.values[moteus.Register.VELOCITY]})"  
    #     + f"({result.values[moteus.Register.TORQUE]})"  
    #     for result in results)
    #     )
    #     positions = [result.values[moteus.Register.POSITION] for result in results]

    #     await asyncio.sleep(0.02)


# Main execution
if __name__ == "__main__":
    # Link 1
    m1 = 0
    l1 = 0


    # Create the robot arm
    arm = create_2dof_arm()
    
    # Create a context for the arm
    context = arm.CreateDefaultContext()
    
    # Example joint positions, velocities, and accelerations
    q = np.array([np.pi/4, np.pi/3])  # positions
    v = np.array([0.1, 0.2])  # velocities
    vd = np.array([0.05, 0.1])  # accelerations
    
    # Calculate inverse dynamics
    tau = calculate_inverse_dynamics(arm, context, q, v, vd)
    
    print("Joint torques:", tau)
    
    # Gravity compensation (set velocities and accelerations to zero)
    tau_gravity = calculate_inverse_dynamics(arm, context, q, np.zeros(2), np.zeros(2))
    
    print("Gravity compensation torques:", tau_gravity)