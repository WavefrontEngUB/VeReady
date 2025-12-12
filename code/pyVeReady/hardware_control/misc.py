import time

def measure_polarization_images(motor_qwp, camera_pymm, n_angles):
    motor_qwp.home()
    time.sleep(1)

    polarization_images_list = []
    for i in range(n_angles):
        image = camera_pymm.snap()
        polarization_images_list.append(image)

        if i == n_angles - 1:
            pass
        else:
            motor_qwp.jog_forward()
            time.sleep(0.4)
    return polarization_images_list