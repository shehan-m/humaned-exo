"""
This example commands multiple servos connected to a pi3hat.  It
uses the .cycle() method in order to optimally use the pi3hat
bandwidth.
"""

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
        
        # For link2, use a parameter that can be updated (assistance from exoskeleton)
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
    # Define parameters for the arm
    m1 = 1.0  # mass of link 1 in kg
    l1 = 0.5  # length of link 1 in meters
    m2 = 0.8  # mass of link 2 in kg
    l2 = 0.4  # length of link 2 in meters
    assistance = 0.2  # initial end mass in kg

    # Initialize the arm model
    arm = Arm(m1, m2, l1, l2, assistance)

    # Set up the transport and servos
    transport = moteus_pi3hat.Pi3HatRouter(
        servo_bus_map={1: [1, 2]},
    )
    servos = {
        servo_id: moteus.Controller(id=servo_id, transport=transport)
        for servo_id in [1, 2]
    }

    # Send initial stop command to all servos
    await transport.cycle([x.make_stop() for x in servos.values()])

    while True:
        # Query the current state from servos
        commands = [
            servos[1].make_position(query=True),
            servos[2].make_position(query=True),
        ]
        results = await transport.cycle(commands)

        # Extract positions and velocities from the results
        q = [results[0].values[moteus.Register.POSITION],
             results[1].values[moteus.Register.POSITION]]
        v = [results[0].values[moteus.Register.VELOCITY],
             results[1].values[moteus.Register.VELOCITY]]
        vd = [0, 0]  # zero acceleration

        # Calculate torques using inverse dynamics
        tau = arm.calculate_inverse_dynamics(q, v, vd)

        # Use the calculated torques as feedforward_torque for each motor
        commands = [
            servos[1].make_position(
                feedforward_torque=tau[0],
                query=True
            ),
            servos[2].make_position(
                feedforward_torque=tau[1],
                query=True
            ),
        ]

        # Send the commands and get responses
        await transport.cycle(commands)

        # Wait 20ms between cycles to prevent watchdog timeout
        await asyncio.sleep(0.02)

if __name__ == '__main__':
    asyncio.run(main())