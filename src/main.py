import asyncio
import math
import moteus
import moteus_pi3hat
import time
import numpy as np

class Arm:
    def __init__(self, m1, m2, l1, l2, initial_end_mass):
        """Initialise the 2-DOF arm with parameters."""
        self.m1 = m1  # mass of link 1
        self.m2 = m2  # mass of link 2
        self.l1 = l1  # length of link 1
        self.l2 = l2  # length of link 2
        self.end_mass = initial_end_mass  # mass at end effector
        
        # Calculate center of mass distances
        self.lc1 = l1/2  # COM of link 1
        self.lc2 = l2/2  # COM of link 2
        
        self.g = 9.81  # gravity constant

    def set_end_mass(self, end_mass):
        """Update the end mass."""
        self.end_mass = end_mass

    def calculate_dynamic_compensation(self, q_rotations, v_rotations):
        """
        Calculate compensation torques including gravity and velocity effects.
        q_rotations: list of joint angles in rotations (arbitrary units)
        v_rotations: list of joint velocities in rotations/time
        Returns: list of torques [shoulder_torque, elbow_torque]
        """
        # Convert rotations to radians
        q = [pos * 2 * np.pi for pos in q_rotations]  # convert to radians
        v = [vel * 2 * np.pi for vel in v_rotations]  # convert to rad/s
        q1, q2 = q
        v1, v2 = v
        
        # Compute trigonometric terms
        s1 = np.sin(q1)
        s2 = np.sin(q2)
        s12 = np.sin(q1 + q2)
        c1 = np.cos(q1)
        c2 = np.cos(q2)
        c12 = np.cos(q1 + q2)
        
        # Gravity compensation terms
        tau1 = (self.m1 * self.g * self.lc1 * c1 +  # link 1 COM
                self.m2 * self.g * (self.l1 * c1 +   # link 2 mass effect
                                  self.lc2 * c12) +
                self.end_mass * self.g * (self.l1 * c1 +  # end mass effect
                                        self.l2 * c12))
        
        tau2 = ((self.m2 * self.lc2 + self.end_mass * self.l2) *
                self.g * c12)

        # Add Coriolis terms
        h = self.m2 * self.l1 * self.lc2 * s2 + self.end_mass * self.l1 * self.l2 * s2
        tau1 -= h * v2 * (v1 + v2)  # Subtract because opposing motion
        tau2 += h * v1 * v1  # Add because assisting motion
        
        return [tau1, tau2]
    
async def main():
    # Define parameters for the arm
    m1 = 0.839  # mass of link 1 in kg
    l1 = 0.265  # length of link 1 in meters
    m2 = 0.203  # mass of link 2 in kg
    l2 = 0.260  # length of link 2 in meters
    assistance = 0.0  # initial end mass in kg

    # Initialize the arm model
    arm = Arm(m1, m2, l1, l2, assistance)

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

                # Add debug printing
                print("\nRaw results from servos:")
                for i, result in enumerate(results):
                    print(f"Servo {i+1} result: {result.values}")

                # Safely extract positions and velocities with error checking
                q = []
                v = []
                for result in results:
                    if moteus.Register.POSITION not in result.values:
                        raise ValueError(f"Position not found in servo results: {result.values}")
                    if moteus.Register.VELOCITY not in result.values:
                        raise ValueError(f"Velocity not found in servo results: {result.values}")
                    
                    q.append(result.values[moteus.Register.POSITION])
                    v.append(result.values[moteus.Register.VELOCITY])

                print(f"Extracted positions: {q}")
                print(f"Extracted velocities: {v}")

                # Calculate torques using inverse dynamics
                tau = arm.calculate_dynamic_compensation(q, v)
                print(f"Calculated torques: {tau}")

                # Use the calculated torques as feedforward_torque for each motor
                commands = [
                    servos[1].make_position(
                        kp_scale=0,
                        kd_scale=0,
                        feedforward_torque=tau[0],
                        query=True
                    ),
                    servos[2].make_position(
                        kp_scale=0,
                        kd_scale=0,
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
            except ValueError as e:
                print(f"ValueError in control loop: {e}")
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
        try:
            await transport.cycle([x.make_stop() for x in servos.values()])
            print("Successfully stopped motors")
        except Exception as e:
            print(f"Error stopping motors: {type(e).__name__}: {e}")

if __name__ == '__main__':
    asyncio.run(main())