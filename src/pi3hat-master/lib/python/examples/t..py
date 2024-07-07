import asyncio
import moteus
import time
import json
async def main():
    qr = moteus.QueryResolution()
    qr._extra = {
        moteus.Register.CONTROL_POSITION:moteus.F32,
        moteus.Register.CONTROL_VELOCITY:moteus.F32,
        moteus.Register.CONTROL_TORQUE:moteus.F32,
        moteus.Register.POSITION_ERROR:moteus.F32,
        moteus.Register.VELOCITY_ERROR:moteus.F32,
        moteus.Register.TORQUE_ERROR:moteus.F32,
    }
    c = moteus.Controller(query_resolution=qr)
    await c.set_stop()

    results = await c.set_position(
            position=None, 
            velocity = 1.0,
            accel_limit = 2.0,
            velocity_limit = 5.0,
            query = True
        )
    '''
    while True:
        current_command = 5#5 if (round(time.time())%2) else -5
        results = await c.set_position(
            position=current_command, 
            velocity = 0.0,
            accel_limit = 2.0,
            velocity_limit = 5.0,
            query = True
        )
    '''
    data = json.loads(str(results))
    print(data)
    
if __name__=="__main__":
    asyncio.run(main())
