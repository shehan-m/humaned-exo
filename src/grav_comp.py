import pinocchio as pin
import numpy as np
from sys import argv

import asyncio
import math
import moteus
import moteus_pi3hat
import time

def inverse_kinematic(q, v, a):
  # Evaluate the derivatives
  pin.computeABADerivatives(model,data,q,v,a)

  # Retrieve the derivatives in data
  ddq_dq = data.ddq_dq # Derivatives of the FD w.r.t. the joint config vector
  ddq_dv = data.ddq_dv # Derivatives of the FD w.r.t. the joint velocity vector
  ddq_dtau = data.Minv # Derivatives of the FD w.r.t. the joint acceleration vector
  
  return ddq_dq, ddq_dv, ddq_dtau

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
  while True:
    dp, dv, dtau = inverse_kinematic(positions, velocity, accel)

    commands = [
        servos[1].make_position(
          feedforward_torque=dtau,
          query=True)
    ]

    results = await transport.cycle(commands)


    print(", ".join(
     f"({result.arbitration_id}) " 
      + f"({result.values[moteus.Register.POSITION]}) " 
      + f"({result.values[moteus.Register.VELOCITY]})"  
      + f"({result.values[moteus.Register.TORQUE]})"  
      for result in results)
    )
    positions = [result.values[moteus.Register.POSITION] for result in results]

    await asyncio.sleep(0.02)


  
if __name__ == '__main__':
  # You should change here to set up your own URDF file or just pass it as an argument of this example.
  urdf_filename = '/models/robot.urdf' if len(argv)<2 else argv[1]
  
  # Load the urdf model
  model = pin.buildModelFromUrdf(urdf_filename)
  print('model name: ' + model.name)
  
  # Create data required by the algorithms
  data = model.createData()
  
  # Set bounds (by default they are undefinded for a the Simple Humanoid model)
  model.lowerPositionLimit = -np.ones((model.nq,1))
  model.upperPositionLimit = np.ones((model.nq,1))

  q = pin.randomConfiguration(model) # joint configuration
  v = np.random.rand(model.nv,1) # joint velocity
  a = np.random.rand(model.nv,1) # joint acceleration
  
  asyncio.run(main())
