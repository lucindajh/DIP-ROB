# In your xArm serial module (adapt from docs)
import xarm
import time

from xarm import Servo

# arm is the first xArm detected which is connected to USB
arm = xarm.Controller('USB')
print('Battery voltage in volts:', arm.getBatteryVoltage())
states = {
        'home': [377, 529, 177, 741, 683, 499],  # Central safe pos (calibrate once)
        'ready_to_grab': [194, 527, 174, 730, 327, 499],  # Above fixed piece spot, gripper open/down
        'ready_to_move': [377, 527, 173, 741, 350, 499]   # Lifted 5-10cm, gripper closed
    }
servos = [xarm.Servo(1), xarm.Servo(2), xarm.Servo(3), xarm.Servo(4), xarm.Servo(5), xarm.Servo(6)]
def set_state(state_name):

    if state_name in states:

        for servo in servos:

            arm.setPosition(servo.servo_id, states[state_name][servo.servo_id-1])
        # Your method: send via TTL serial
        print(f"Set to {state_name}")
        #
    else:
        print("Invalid state")

def reset_state(state_name):
    if state_name in states:
        states[state_name] = get_joints()

def get_joints():
    positions = []
    for servo in servos:
        positions.append(arm.getPosition(servo))
    return positions

        # Returns list of 6 floats + voltages

def check_grip_success(threshold=0.5):  # Via end-effector feedback
    positions, voltages = arm.read_positions_and_voltages()
    # Simple: Grip success if voltage spike on gripper servo (j6?)
    return any(v > threshold for v in voltages[-1:])

def disengage():
    for servo in servos:
        arm.servoOff(servo.servo_id)

# set_state('ready_to_grab')
print(get_joints())

# time.sleep(5)
# disengage()