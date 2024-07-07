import asyncio
import math
import moteus

async def main():
    transport = moteus.Fdcanusb()
    c1 = moteus.Controller(id = 1)
    c2 = moteus.Controller(id = 2)

    while True:
        print(await transport.cycle([
          c1.make_position(position=math.nan, query=True),
          c2.make_position(position=math.nan, query=True),
        ]))

asyncio.run(main())