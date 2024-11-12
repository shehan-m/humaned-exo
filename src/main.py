import asyncio
import moteus
import moteus_pi3hat
import numpy as np
import pinocchio as pin
from typing import List, Tuple

class ArmDynamics:
    def __init__(self, m1, m2, l1, l2, com1, com2):
        """Initialize the 2-DOF planar arm with Pinocchio model."""
        self.model = self.build_model(m1, m2, l1, l2, com1, com2)
        self.data = self.model.createData()
        
        # Store frame ID for end-effector computations
        self.ee_frame_id = self.model.getFrameId("end_effector")

    def build_model(self, m1, m2, l1, l2, com1, com2):
        """Build the 2-DOF planar arm model following the provided structure."""
        model = pin.Model()
        
        # Constants
        kFudge = 0.95  # Safety factor for inertias
        
        # Link transformations (from joint to COM)
        Tlink1 = pin.SE3(pin.SE3.Matrix3.Identity(), np.array([com1, 0, 0]))  # CHANGE THIS TO ACTUAL
        Tlink2 = pin.SE3(pin.SE3.Matrix3.Identity(), np.array([com2, 0, 0]))  # CHANGE THIS TO ACTUAL
        
        # Inertias (mass, COM position, rotational inertia)
        Ilink1 = pin.Inertia(
            kFudge * m1,  # Mass with safety factor
            Tlink1.translation,
            pin.Inertia.Matrix3.Identity() * 0.001
        )
        Ilink2 = pin.Inertia(
            kFudge * m2,  # Mass with safety factor
            Tlink2.translation,
            pin.Inertia.Matrix3.Identity() * 0.001
        )

        # Improve the above section by adding correct rotational inertias:
        # Ilink1 = pin.Inertia(
        #     kFudge * m1,
        #     Tlink1.translation,
        #     pin.Inertia.Matrix3(
        #         Ixx, 0, 0,
        #         0, Iyy, 0,
        #         0, 0, Izz
        #     )
        # )
        
        # Joint limits
        qmin = np.array([-5.0])  # Position limits in radians
        qmax = np.array([5.0])
        vmax = np.array([20])   # Velocity limits in rad/s
        taumax = np.array([11]) # Torque limits in N⋅m
        
        # Add joints and bodies
        idx = 0  # Start with root joint
        
        # Link 1
        joint1_placement = pin.SE3(pin.SE3.Matrix3.Identity(), np.array([0, 0, 0]))
        idx = model.addJoint(
            idx,
            pin.JointModelRX(),  # Revolute joint around X (for planar motion in Y-Z plane)
            joint1_placement,
            "joint1",
            taumax,
            vmax,
            qmin,
            qmax
        )
        model.appendBodyToJoint(idx, Ilink1)
        model.addJointFrame(idx)
        model.addBodyFrame("body1", idx)
        
        # Link 2
        joint2_placement = pin.SE3(
            pin.SE3.Matrix3.Identity(),
            np.array([l1, 0, 0])  # Full length of link 1
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
        model.appendBodyToJoint(idx, Ilink2)
        model.addJointFrame(idx)
        model.addBodyFrame("body2", idx)
        
        # Add end-effector frame
        ee_placement = pin.SE3(
            pin.SE3.Matrix3.Identity(),
            np.array([l2, 0, 0])  # Full length of link 2
        )
        model.addFrame(
            pin.Frame(
                "end_effector",
                idx,  # Attach to last joint
                0,
                ee_placement,
                pin.FrameType.OP_FRAME
            )
        )

        return model

    def compute_dynamics(
        self, 
        q: List[float],
        v: List[float], 
        end_wrench: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute inverse dynamics with variable end-effector wrench.
        
        Args:
            q: Joint positions in rotations
            v: Joint velocities in rotations/s
            end_wrench: 6D spatial force (force/torque) at end-effector [fx,fy,fz,tx,ty,tz]
        
        Returns:
            tau: Joint torques accounting for gravity and end-effector wrench
        """
        # Convert rotations to radians
        q_rad = np.array([pos * 2 * np.pi for pos in q])
        v_rad = np.array([vel * 2 * np.pi for vel in v])
        
        # Zero acceleration (we only want gravity comp + external forces)
        a_rad = np.zeros_like(q_rad)

        # Compute the dynamics using RNEA
        tau_gravity = pin.rnea(self.model, self.data, q_rad, v_rad, a_rad)
        
        # Get Jacobian at end-effector
        pin.computeJointJacobians(self.model, self.data, q_rad)
        J = pin.getFrameJacobian(
            self.model,
            self.data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        # Calculate joint torques from end-effector wrench: τ = J^T * h_e
        tau_ee = J.T @ end_wrench
        
        # Total torque is gravity compensation plus end-effector contribution
        tau_total = tau_gravity + tau_ee
        
        return tau_total

    
async def main():
    # Starting offset from zero
    offset = [0.05, 0.1] # TODO: Change this with actual

    # Define parameters for the arm
    m1 = 0.839  # mass of link 1 in kg
    l1 = 0.310  # length of link 1 in meters
    m2 = 0.203  # mass of link 2 in kg
    l2 = 0.265  # length of link 2 in meters
    com1 = 0.225 # Centre of mass of link 1 in meters
    com2 = 0.130
    end_mass = 0.0  # initial end mass in kg

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

                # End-effector force due to gravity
                end_force = np.array([0, 0, -end_mass * 9.81, 0, 0, 0])

                # Calculate torques using inverse dynamics
                tau = arm.compute_dynamics(q, v, end_force)

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