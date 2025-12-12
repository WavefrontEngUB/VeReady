import sys, os, time
import clr

clr.AddReference('System')
clr.AddReference(r'C:\Program Files\Thorlabs\Elliptec\Thorlabs.Elliptec.ELLO_DLL.dll')
from System import Decimal as SysDecimal
from Thorlabs.Elliptec.ELLO_DLL import *


class AddressedDevice:
    """Class to represent an addressed Elliptec device."""

    def __init__(self, addressed_device):
        self.addressed_device = addressed_device
        self.device_info = addressed_device.DeviceInfo

    def print_device_info(self):
        """Prints detailed information about the device."""
        print("DEVICE INFO:")
        for info in self.device_info.Description():
            print(info)

    def set_jog_step(self, jog_step):
        self.addressed_device.SetJogstepSize(SysDecimal(jog_step))

    def set_home_offset(self, home_offset):
        self.addressed_device.SetHomeOffset(SysDecimal(home_offset))

    def jog_forward(self):
        """Moves the device forward."""
        self.addressed_device.JogForward()

    def jog_backward(self):
        """Moves the device backward."""
        self.addressed_device.JogBackward()

    def home(self):
        """Moves the device to its home position."""
        self.addressed_device.Home(ELLBaseDevice.DeviceDirection.Clockwise)


def init_and_scan_devices(com_port):
    """
    Scans for available Elliptec devices, prints their information,
    and returns a list of addressed devices.
    """

    print("Scanning for available devices...")

    # Connect to device
    ELLDevicePort.Connect(com_port)

    # Define byte address range
    min_address = "0"
    max_address = "F"

    # Scan for devices
    ellDevices = ELLDevices()
    devices = ellDevices.ScanAddresses(min_address, max_address)

    addressed_devices = []
    ordinals = ['FIRST', 'SECOND', 'THIRD', 'FOURTH']

    for i, device in enumerate(devices):
        if ellDevices.Configure(device):
            print(f'\n{ordinals[i]} DEVICE')
            addressed_device = ellDevices.AddressedDevice(device[0])
            device_info = addressed_device.DeviceInfo
            for info in device_info.Description():
                print(info)

            # Store the addressed device in the list
            addressed_devices.append(addressed_device)

    print("\nScan complete.")
    return addressed_devices  # Return the list of addressed devices
