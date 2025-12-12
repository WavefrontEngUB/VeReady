import imagingcontrol4 as ic4
import cv2
import numpy as np
from pyVeReady.measure_scripts.motor_control_python import *

print('JOGSTEPS NEED TO BE SET ')
# Set Up motors
adressed_motors, motors_info = init_elliptec_motor('COM3', jog_test = True)

motor_qwp, info_motor_qwp = adressed_motors[0], motors_info[0]
motor_vortex, info_motor_vortex = adressed_motors[1], motors_info[1]

# Set Up Camera
ic4.Library.init(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR)

device_list = ic4.DeviceEnum.devices()
for i, dev in enumerate(device_list):
    print(f"[{i}] {dev.model_name} ({dev.serial}) [{dev.interface.display_name}]")
print(f"Selected Device: 0", end="")
selected_index = 0
dev_info = device_list[selected_index]

grabber = ic4.Grabber()
grabber.device_open(dev_info)

cv2.namedWindow("display")
cv2.waitKey(0)

sink = ic4.SnapSink()
grabber.stream_setup(sink)

# Set Exposure
while True:
    print("Press any key on the Image window to continue...")
    cv2.waitKey(0)
    print('Set Exposure Value')
    exposure_input_ms = float(input("Enter Exposure Value (ms): "))
    if exposure_input_ms == 0:
        cv2.destroyWindow("display")
        break
    grabber.device_property_map.set_value(ic4.PropId.EXPOSURE_TIME, exposure_input_ms * 1e3)

    buffer = sink.snap_single(1000)
    im_array = buffer.numpy_wrap()
    cv2.imshow("display", im_array)

motor_qwp.Home(ELLBaseDevice.DeviceDirection.Clockwise)
motor_vortex.Home(ELLBaseDevice.DeviceDirection.Clockwise)
time.sleep(10)

n_vortex_rotations = 40
n_qwp_rotations = 8

polarization_images = np.zeros((480, 744, int(n_vortex_rotations * n_qwp_rotations)))
slice = 0
for ii in range(n_vortex_rotations):
    print(ii)
    for jj in range(n_qwp_rotations):
        buffer = sink.snap_single(1000)
        im_array = buffer.numpy_wrap()
        polarization_images[:,:,slice] = im_array[:,:,0]

        if jj == (n_qwp_rotations - 1):
            motor_qwp.Home(ELLBaseDevice.DeviceDirection.Clockwise)
            time.sleep(2)
        else:
            motor_qwp.JogForward()
            time.sleep(1)

        slice = slice + 1

    motor_vortex.JogForward()
    time.sleep(1)

saving_name = input('Enter Saving Name')
np.save(saving_name, polarization_images)