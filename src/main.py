import asyncio
import moteus
import moteus_pi3hat
import numpy as np
import pinocchio as pin
from typing import List, Tuple
import math

class ArmDynamics:
    def __init__(self, m1, m2, l1, l2, com1, com2):
        """Initialize the 2-DOF planar arm with Pinocchio model."""
        self.model = self.build_model(m1, m2, l1, l2, com1, com2)
        self.data = self.model.createData()

    def build_model(self, m1, m2, l1, l2, com1, com2):
        """Build the 2-DOF planar arm model following the provided structure."""
        model = pin.Model()
        
        # Constants
        kFudge = 0.99  # Safety factor for inertias
        
        # Link transformations (from joint to COM)
        Tlink1 = pin.SE3(pin.utils.eye(3), np.array([com1, 0, 0]))
        Tlink2 = pin.SE3(pin.utils.eye(3), np.array([com2, 0, 0]))
        
        # Inertias (mass, COM position, rotational inertia)
        Ilink1 = pin.Inertia(
            kFudge * m1,
            Tlink1.translation,
            pin.utils.eye(3) * 0.001
        )
        Ilink2 = pin.Inertia(
            kFudge * m2,
            Tlink2.translation,
            pin.utils.eye(3) * 0.001
        )
        
        # Joint limits
        qmin = np.array([-5.0])
        qmax = np.array([5.0])
        vmax = np.array([20])
        taumax = np.array([11])
        
        # Add joints and bodies
        idx = 0
        
        # Link 1
        joint1_placement = pin.SE3(pin.utils.eye(3), np.array([0, 0, 0]))
        idx = model.addJoint(
            idx,
            pin.JointModelRX(),
            joint1_placement,
            "joint1",
            taumax,
            vmax,
            qmin,
            qmax
        )
        model.appendBodyToJoint(idx, Ilink1, pin.SE3.Identity())
        model.addJointFrame(idx)
        model.addBodyFrame("body1", idx, pin.SE3.Identity(), -1)
        
        # Link 2
        joint2_placement = pin.SE3(
            pin.utils.eye(3),
            np.array([l1, 0, 0])
        )
        idx = model.addJoint(
            idx,
            pin.JointModelRX(),
            joint2_placement,
            "joint2",
            taumax,
            vmax,
            qmin,
            qmax
        )
        model.appendBodyToJoint(idx, Ilink2, pin.SE3.Identity())
        model.addJointFrame(idx)
        model.addBodyFrame("body2", idx, pin.SE3.Identity(), -1)
        
        return model

    def compute_dynamics(self, q: List[float], v: List[float]) -> np.ndarray:
        """Compute inverse dynamics without end-effector contribution.
        
        Args:
            q: Joint positions in rotations
            v: Joint velocities in rotations/s
        
        Returns:
            tau: Joint torques accounting only for gravity
        """
        # Convert rotations to radians
        q_rad = np.array([pos * 2 * np.pi for pos in q])
        v_rad = np.array([vel * 2 * np.pi for vel in v])
        
        # Zero acceleration (we only want gravity compensation)
        a_rad = np.zeros_like(q_rad)

        # Compute the dynamics using RNEA for gravity compensation only
        tau_gravity = pin.rnea(self.model, self.data, q_rad, v_rad, a_rad)
        
        return tau_gravity

    
async def main():
    # Starting offset from zero
    offset = [-0.0272064208984375, -0.057861328125]

    # Define parameters for the arm
    m1 = 0.839  # mass of link 1 in kg
    l1 = 0.310  # length of link 1 in meters
    m2 = 0.203  # mass of link 2 in kg
    l2 = 0.265  # length of link 2 in meters
    com1 = 0.225 # Centre of mass of link 1 in meters
    com2 = 0.130

    # Initialize the arm model
    arm = ArmDynamics(m1, m2, l1, l2, com1, com2)

    try:
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
        print("Successfully initialized servos")

        print("Starting dynamic compensation loop...")
        
        while True:
            try:
                # Query the current state from servos
                commands = [
                    servos[1].make_position(query=True),
                    servos[2].make_position(query=True),
                ]
                results = await transport.cycle(commands)

                print("\nRaw results from servos:")
                for i, result in enumerate(results):
                    print(f"Servo {i+1} result: {result.values}")

                # Safely extract positions and velocities with error checking
                q = [results[0].values[moteus.Register.POSITION], results[1].values[moteus.Register.POSITION]]
                v = [results[0].values[moteus.Register.VELOCITY], results[1].values[moteus.Register.VELOCITY]]

                # Calculate torques using inverse dynamics for gravity compensation
                tau = arm.compute_dynamics(q, v)

                # Use the calculated torques as feedforward_torque for each motor
                commands = [
                    servos[1].make_position(
                    	position=math.nan,
                    	velocity=math.nan,
                        kp_scale=0.0,
                        kd_scale=0.0,
                        feedforward_torque=tau[0],
                        query=True
                    ),
                    servos[2].make_position(
                    	position=math.nan,
                    	velocity=math.nan,
                        kp_scale=0.0,
                        kd_scale=0.0,
                        feedforward_torque=tau[1],
                        query=True
                    ),
                ]

                # Send the commands and get responses
                await transport.cycle(commands)

            except KeyError as e:
                print(f"KeyError accessing motor data: {e}")
                print("Available registers:", results[0].values.keys() if results else "No results")
                break
            except Exception as e:
                print(f"Unexpected error in control loop: {type(e).__name__}: {e}")
                break

            # Wait 20ms between cycles to prevent watchdog timeout
            await asyncio.sleep(0.02)

    except Exception as e:
        print(f"Error during initialization: {type(e).__name__}: {e}")
    finally:
        # Ensure we always try to stop the motors
        await transport.cycle([x.make_stop() for x in servos.values()])
        print("Successfully stopped motors")

if __name__ == '__main__':
    asyncio.run(main())
