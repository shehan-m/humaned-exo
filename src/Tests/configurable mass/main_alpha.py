import asyncio
import moteus
import moteus_pi3hat
import numpy as np
import pinocchio as pin
import os
import math
import logging
import pickle
import time

class ArmDynamics:
    def __init__(self, urdf_path):
        """Initialize the 2-DOF planar arm using a URDF file."""
        cache_path = urdf_path.replace(".urdf", ".pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = pin.buildModelFromUrdf(urdf_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.model, f)

        self.data = self.model.createData()
        self.zero_acc = np.zeros(2)

        # Identify end-effector joint
        self.end_effector_id = self.model.njoints - 1

        # Preallocate external force vector (initialize with zeros)
        self.fext = pin.StdVec_Force()
        self.fext.extend([pin.Force.Zero() for _ in range(self.model.njoints)])

    def compute_dynamics(self, q: np.ndarray, v: np.ndarray, mass: float) -> np.ndarray:
        """Compute joint torques for gravity compensation using URDF model.
        
        Args:
            q: Joint positions in radians
            v: Joint velocities in radians/s
        
        Returns:
            tau: Joint torques accounting only for gravity
        """
        
        # Compute forward kinematics to get joint transforms
        pin.forwardKinematics(self.model, self.data, q, v, self.zero_acc)
        
        # Only update the transform of the end-effector (not all joints)
        pin.updateFramePlacement(self.model, self.data, self.end_effector_id)

        # Extract end-effector transform
        ee_placement = self.data.oMi[self.end_effector_id]

        # Gravity force in world frame
        gravity_vector_world = np.array([0, -9.81 * mass, 0])

        # Transform force to local frame using matrix components
        R = ee_placement.rotation.T  # Transpose of rotation matrix
        gravity_vector_local = np.array([
            R[0, 0] * gravity_vector_world[0] + R[0, 1] * gravity_vector_world[1] + R[0, 2] * gravity_vector_world[2],
            R[1, 0] * gravity_vector_world[0] + R[1, 1] * gravity_vector_world[1] + R[1, 2] * gravity_vector_world[2],
            R[2, 0] * gravity_vector_world[0] + R[2, 1] * gravity_vector_world[1] + R[2, 2] * gravity_vector_world[2]
        ])  # Faster than `@` matrix multiply

        # Update only the end-effector joint in fext (reuse memory)
        self.fext[self.end_effector_id].linear = gravity_vector_local
        self.fext[self.end_effector_id].angular = np.zeros(3)

        # Compute torques with external forces
        tau_gravity = pin.rnea(self.model, self.data, q, v, self.zero_acc, self.fext)
        
        return tau_gravity

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Path to the URDF file
    urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm.urdf")
    
    # Initialize the arm model
    arm = ArmDynamics(urdf_path)

    # Precomputed constant and offsets
    TWO_PI = 2 * math.pi    
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
        stop_commands = [servo.make_stop() for servo in servos.values()]
        await transport.cycle(stop_commands)
        
        logging.info("Successfully initialized servos")
        logging.info("Starting dynamic compensation loop...")

        # Pre-allocate numpy arrays for positions and velocities
        q = np.zeros(2)
        v = np.zeros(2)

        target_cycle_time = 0.02  # 20ms cycle time

        # Main control loop
        while True:
            loop_start = time.monotonic()
            try:
                # Query current state from servos with a reduced timeout (e.g., 300ms)
                query_commands = [
                    servo.make_position(query=True) for servo in servos.values()
                ]
                results = await asyncio.wait_for(transport.cycle(query_commands), timeout=0.3)

                # Extract positions and velocities
                try:
                    pos0 = results[0].values[moteus.Register.POSITION]
                    pos1 = results[1].values[moteus.Register.POSITION]
                    vel0 = results[0].values[moteus.Register.VELOCITY]
                    vel1 = results[1].values[moteus.Register.VELOCITY]
                except KeyError as e:
                    logging.error(f"KeyError accessing motor data: {e}. Results: {results}")
                    continue  # Skip this cycle and try again

                # Update joint positions/velocities (using preallocated arrays)
                q[0] = (pos0 - offset[0]) * TWO_PI
                q[1] = (pos1 - offset[1]) * TWO_PI
                v[0] = vel0 * TWO_PI
                v[1] = vel1 * TWO_PI

                # Calculate torques using inverse dynamics for gravity compensation
                tau = arm.compute_dynamics(q, v)

                # Use the calculated torques as feedforward torque for each motor
                torque_commands = [
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
                await transport.cycle(torque_commands)

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

            # Calculate loop duration and adjust sleep to maintain a consistent 20ms cycle
            elapsed = time.monotonic() - loop_start
            sleep_time = target_cycle_time - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                logging.warning(f"Loop overran expected cycle time by {-sleep_time:.4f} seconds.")

    except Exception as e:
        logging.error(f"Error during initialization: {type(e).__name__}: {e}")
    finally:
        # Ensure we always try to stop the motors
        await transport.cycle([x.make_stop() for x in servos.values()])
        await transport.close()
        logging.info("Successfully stopped motors and closed transport")

if __name__ == '__main__':
    asyncio.run(main())
