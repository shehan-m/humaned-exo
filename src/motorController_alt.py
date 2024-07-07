import asyncio
import math
import moteus
import moteus_pi3hat
import time
import keyboard

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
      for result in results)
    )
  
  positions = [result.values[moteus.Register.POSITION] for result in results]
  p = positions[0]
  dR = 0.01
  print("\nStarting loop")
  while True:
    if keyboard.is_pressed("w"):
      #dR = 0.01
      p = positions[0]+dR
    elif keyboard.is_pressed("s"):
      p = positions[0]-dR
    elif keyboard.is_pressed("x"):
      p = positions[0]
    elif keyboard.is_pressed("Esc"):
      break

    commands = [
        servos[1].make_position(
          position=p, 
          maximum_torque=0.1,
          query=True)
    ]

    results = await transport.cycle(commands)

    print(", ".join(
      f"({result.arbitration_id}) " 
      + f"({result.values[moteus.Register.POSITION]}) " 
      + f"({result.values[moteus.Register.VELOCITY]})"  
      for result in results)
    )
    positions = [result.values[moteus.Register.POSITION] for result in results]

    await asyncio.sleep(0.02)


  
if __name__ == '__main__':
  asyncio.run(main())