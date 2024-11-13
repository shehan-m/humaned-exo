import asyncio
import moteus
import moteus_pi3hat
import numpy as np
import pinocchio as pin
from os.path import join, abspath, dirname
import math

class ArmDynamics:
    def __init__(self, urdf_path):
        """Initialize the 2-DOF planar arm using a URDF file."""
        # Load the URDF model
        self.model = pin.buildModelFromUrdf(urdf_path)
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

async def main():
    # Path to the URDF file
    urdf_path = join(dirname(abspath(__file__)), "arm.urdf")
    
    # Initialize the arm model
    arm = ArmDynamics(urdf_path)

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

                # Extract positions and velocities
                q = np.array([
                    results[0].values[moteus.Register.POSITION] * 2 * np.pi,
                    results[1].values[moteus.Register.POSITION] * 2 * np.pi
                ])
                v = np.array([
                    results[0].values[moteus.Register.VELOCITY] * 2 * np.pi,
                    results[1].values[moteus.Register.VELOCITY] * 2 * np.pi
                ])

                # Calculate torques using inverse dynamics for gravity compensation
                tau = arm.compute_dynamics(q, v)

                # Use the calculated torques as feedforward torque for each motor
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
