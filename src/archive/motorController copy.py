import asyncio
import math
import moteus
import moteus_pi3hat
import time

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

  await transport.cycle([x.make_stop() for x in servos.values()])

  while True:
    now = time.time()
    """servos[1].make_position(
        position=math.nan,
        velocity = 3,
        query=True)"""
    commands = [
        servos[1].make_position(
          position=0, 
          maximum_torque=0.1,
          query=True)
    ]

    results = await transport.cycle(commands)

    print(", ".join(
      f"({result.arbitration_id} " 
      + f"({result.values[moteus.Register.POSITION]} " 
      + f"({result.values[moteus.Register.VELOCITY]}"  
      for result in results)
    )

    await asyncio.sleep(0.02)


  
if __name__ == '__main__':
  asyncio.run(main())