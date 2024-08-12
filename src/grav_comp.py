import asyncio
import math
import moteus
import moteus_pi3hat

MASS_KG = 0.26  # Actual value is 0.3 kg
LINK_LENGTH = 0.15  # in meters
GRAVITY = 9.8  # m/s^2

async def main():
    transport = moteus_pi3hat.Pi3HatRouter(
        servo_bus_map={
            1: [1],  # Assuming controller ID 1 is on bus 1
        }
    )
    c = moteus.Controller(id=1, transport=transport)

    # Configure the position mode command
    cmd = moteus.Controller.make_position(
        position=math.nan,
        velocity=0,
        kp_scale=0,
        kd_scale=0,
        feedforward_torque=0,
        query=True
    )

    missed_replies = 0
    status_count = 0
    STATUS_PERIOD = 100

    while True:
        try:
            result = await c.set_position(**cmd)
            state = result.values

            # Calculate gravity compensation torque
            position = state.position
            command_torque = math.sin(position * 2 * math.pi) * MASS_KG * LINK_LENGTH * GRAVITY

            # Update the command
            cmd['feedforward_torque'] = command_torque

            # Reset missed replies counter
            missed_replies = 0

            # Print status periodically
            status_count += 1
            if status_count >= STATUS_PERIOD:
                print(f"Mode: {state.mode:2d}  position: {position:6.3f}  "
                      f"cmd_torque: {command_torque:6.3f}  temp: {state.temperature:4.1f}  ", 
                      end='\r', flush=True)
                status_count = 0

        except moteus.CommandError:
            missed_replies += 1
            if missed_replies > 3:
                print("\n\nMotor timeout!")
                break

        await asyncio.sleep(0.01)  # Similar to usleep(10000) in C++

    print("Entering fault mode!")
    while True:
        await c.set_brake()
        await asyncio.sleep(0.05)

if __name__ == "__main__":
    asyncio.run(main())