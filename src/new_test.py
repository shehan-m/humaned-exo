import moteus
motor = moteus.Controller()
motor.set_position(position=0.5, accel_limit=2.0)