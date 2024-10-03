import asyncio
import math
import moteus
import moteus_pi3hat
import time
import keyboard
import math

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
    i = 0
    while True:
        i += 0.000001
        s = math.sin(i)

        commands = [
            servos[1].make_position(
            position=s, 
            maximum_torque=0.5,
            query=True)
        ]

        results = await transport.cycle(commands)


        #print(", ".join(
        # f"({result.arbitration_id}) " 
        #+ f"({result.values[moteus.Register.POSITION]}) " 
        #+ f"({result.values[moteus.Register.VELOCITY]})"  
        #+ f"({result.values[moteus.Register.TORQUE]})"  
        #f"{p}"
        #for result in results)
        #)
        positions = [result.values[moteus.Register.POSITION] for result in results]

        print(i,s,positions[0])

        await asyncio.sleep(0.02)


  
if __name__ == '__main__':
  asyncio.run(main())