import asyncio
import moteus
import moteus_pi3hat
import numpy as np
import pinocchio as pin
from os.path import join, abspath, dirname, exists
import math
import logging
import pickle

class ArmDynamics:
    def __init__(self, urdf_path):
        """Initialise the 2-DOF planar arm using a URDF file."""
        # Cache the model to improve performance
        cache_path = urdf_path.replace(".urdf", ".pkl")
        if exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = pin.buildModelFromUrdf(urdf_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.model, f)

        self.data = self.model.createData()

        # Preallocate zero acceleration vector
        self.zero_acc = np.zeros(2)

    def compute_dynamics(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute joint torques for gravity compensation using URDF model.
        
        Args:
            q: Joint positions in radians
            v: Joint velocities in radians/s
        
        Returns:
            tau: Joint torques accounting only for gravity
        """

        # Compute torques using RNEA for gravity compensation
        tau_gravity = pin.rnea(self.model, self.data, q, v, self.zero_acc)
        
        return tau_gravity

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Path to the URDF file
    urdf_path = join(dirname(abspath(__file__)), "arm.urdf")
    
    # Initialize the arm model
    arm = ArmDynamics(urdf_path)

    offset = [0, 0]

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
        
        logging.info("Successfully initialized servos")
        logging.info("Starting dynamic compensation loop...")
        
        while True:
            try:
                # Query the current state from servos
                commands = [
                    servo.make_position(query=True) for servo in servos.values()
                ]
                results = await asyncio.wait_for(transport.cycle(commands), timeout=0.5)

                #for i, result in enumerate(results):
                #    print(f"Servo {i+1} result: {result.values}")

                # Extract positions and velocities
                q = np.array([
                    (results[0].values[moteus.Register.POSITION] - offset[0]) * 2 * np.pi,
                    (results[1].values[moteus.Register.POSITION] - offset[1]) * 2 * np.pi
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

            except asyncio.TimeoutError:
                logging.error("Transport cycle timed out.")
                break
            except KeyError as e:
                logging.error(f"KeyError accessing motor data: {e}")
                logging.error(f"Available registers: {results if results else 'No results'}")
                break
            except Exception as e:
                logging.error(f"Unexpected error in control loop: {type(e).__name__}: {e}")
                break

            # Wait 20ms between cycles to prevent watchdog timeout
            await asyncio.sleep(0.02)

    except Exception as e:
        logging.error(f"Error during initialization: {type(e).__name__}: {e}")
    finally:
        # Ensure we always try to stop the motors
        await transport.cycle([x.make_stop() for x in servos.values()])
        await transport.close()
        logging.info("Successfully stopped motors and closed transport")

if __name__ == '__main__':
    asyncio.run(main())
