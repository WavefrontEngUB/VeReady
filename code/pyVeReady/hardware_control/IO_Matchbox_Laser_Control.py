import serial
import glob
import sys
import time

class IntegratedOpticsMatchbox:
    '''
    Object for an Integrated Optics Matchbox Laser. It operates via Serial string commands.
    Code obtained from https://github.com/iancynk/IOmatchbox/tree/main
    '''
    # set this to True when you need a more output
    # not really implemented
    DEBUG = False

    def __init__(self, port='', DEBUG=False):
        self.ser = None
        # open serial connection
        if port:
            # with port if port is given
            self.openlaser(port=port)
        else:
            # otherwise just pick the first found port
            self.openlaser()
        # more output info
        if DEBUG: self.DEBUG = True

    def __str__(self):
        return f"Is a serial instance of a IO laser."
        self.get_info()

    # --------------------------------------------------------------------------
    # communication functions

    def check_reply(self, reply):
        """interpret reply/errorcode according to manual"""
        match reply:
            case '<ACK>':
                print('acknowledged')
            case '<ERR 0>':
                print('0 - error name not assigned yet')
            case '<ERR 1>':
                print('1 - command forbidden for current access level')
            case '<ERR 2>':
                print('2 - laser already on or making ramp-up')
            case '<ERR 3>':
                print('3 - laser busy, task is not complete please wait for 1 s and try again')
            case '<ERR 4>':
                print('4 - arguments out of range')
            case '<ERR 5>':
                print('5 - unknown command')
            case '<ERR 6>':
                print('6 - laser must be enabled to execute this command')
            case _:
                print('unknown reply:', reply)

    # --------------------------------------------------------------------------
    # serial functions

    def openlaser(self, port=''):
        """create a serial connection with the recommended parameters
        if no port is given the function will try all available serial ports
        and check whether the connected device has an ID like an IO laser.
        """
        self.ser = serial.Serial()
        self.ser.baudrate = 115200
        self.ser.bytesize = serial.EIGHTBITS
        self.ser.parity = serial.PARITY_NONE
        self.ser.stopbits = serial.STOPBITS_ONE
        self.ser.timeout = 2
        # self.ser.rtscts = True # enable hardware (TRS/CTS) flow control
        if not port:
            # find available ports depending on operating system
            if sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
                # look for the id of the UART controller
                available_ports = glob.glob('/dev/serial/by-id/*CP2102N*')
                if available_ports == '':
                    # look for all USB ports
                    available_ports = glob.glob('/dev/ttyUSB*')
                print('available ports:', available_ports)
            elif sys.platform.startswith('win'):
                available_ports = ['COM%s' % (i + 1) for i in range(256)]
            else:
                raise EnvironmentError('Unsupported platform')
        else:
            available_ports = [port]

        if not available_ports:
            print('no serial port selected, aborting')
            self.ser = None
            return

        # try to open the ports until one works
        for port in available_ports:
            try:
                print('opening port', port)
                self.ser.port = port
                self.ser.open()
                time.sleep(0.1)
            except:
                print('failed at port', port)
                continue
            # check if the selected port has an IO laser by querying the ID
            _ = self.ser.write("NM?".encode())
            # check if the ID contains numbers at [1:4]
            try:
                reply = self.ser.readline().decode('utf-8').strip()
                int(reply[1:4])
                print('connected to:', reply[1:-1])
                break
            except:
                self.ser.close()
                time.sleep(0.1)
                print('not a IO CW laser')
                pass

        if self.DEBUG:
            if self.ser.is_open:
                print('port', self.ser.port, 'opened')
        return

    def port_is_open(self):
        """check whether serial port is open
        returns false if port is closed or not a serial port
        """
        try:
            if not self.ser.is_open:
                print('serial port not open')
                return False
        except AttributeError:
            print('no serial stage connected, ignoring command')
            return False
        return True

    def closelaser(self):
        """close serial connection"""
        if not self.port_is_open(): return
        self.ser.close()
        if not self.ser.is_open:
            print('connection closed')

    # --------------------------------------------------------------------------
    # possible commands (from MB IO user manual)

    cmds = {
        "product_ID": "ID?",  # return product ID (6 digits)
        "product_code": "NM?",  # returns laser name (product code)
        "get_settings": "r s",  # receive settings
        "get_readings": "r r",  # receive readings
        "get_om": "r m",  # receive operating mode (APC/ACC)
        "get_info": "r i",  # receive laser information
        "get_optime": "r t",  # receive operating hours/switching times
        "get_access_level": "r l",  # receive access level
        "set_access_level": "c u",  # change access level (requires number and code)
        "set_crystal_temp": "c 1",  # set crystal temp in centidegC (eg 2550))
        "set_diode_temp": "c 2",  # set laser diode temp in centidegC (eg 2550)
        "set_diode_current": "c 3",  # set laser diode current in mA (requires current)
        "set_opt_power": "c 4",  # set optical power (requires power in mW, only if feedback diode integrated)
        "set_feedback_DAC": "c 6",  # set feedback DAC value (value 0..8191)
        "set_fan_temp": "c f",  # set fan temperature in centidegC
        "enable_autostart": "c a",  # enable/disable autostart after power on (1 or 0)
        "enable_laser": "e",  # start/stop laser (1/0)
        "laser_warmup": "e 2",  # enable warm-up of laser (recommended for longevity)
        "save_changes": "f s"  # save changes
    }

    # --------------------------------------------------------------------------
    # read out settings with commands

    def get_ID(self):
        """get module ID"""
        _ = self.ser.write(self.cmds["product_ID"].encode())
        reply = self.ser.readline().decode('utf-8').strip()
        print('product ID:', reply[1:-1])

    def get_productcode(self):
        """get module product code"""
        _ = self.ser.write(self.cmds["product_code"].encode())
        reply = self.ser.readline().decode('utf-8').strip()
        print('product code:', reply[1:-1])

    def get_settings(self, output=False):
        """receive laser settings
        if output=True, will spill the whole string with info, otherwise only returns it
        """
        _ = self.ser.write(self.cmds["get_settings"].encode())
        reply = self.ser.readline().decode('utf-8').strip()
        if output:
            print('Settings: crystal set temp, laser diode set temperature, \
    laser diode set current, feedback DAC set value, optical power set value, \
    laser diode current limit (mA), autostart mode, access level, fan set temp')
            print(reply)
            return
        else:
            return reply

    def get_readings(self, output=False):
        """receive laser readings
        if output=True, will spill the whole string with info, otherwise only returns it
        """
        _ = self.ser.write(self.cmds["get_readings"].encode())
        reply = self.ser.readline().decode('utf-8').strip()
        if output:
            print('Readings:laser diode temperature, crystal temperature (negative if there is no crystal), \
    laser base temperature, laser diode current, crystal TEC load, \
    laser diode TEC load, system status, fan load, input voltage')
            print(reply)
            return
        else:
            return reply

    def get_om(self):
        """receive operation mode (APC/ACC)"""
        _ = self.ser.write(self.cmds["get_om"].encode())
        reply = self.ser.readline().decode('utf-8').strip()
        print('operation mode:', reply)

    def get_info(self, output=False):
        """receive laser information
        if output=True, will spill the whole string with info, otherwise only returns it
        """
        _ = self.ser.write(self.cmds["get_info"].encode())
        # output comes in five lines, combine it to one list
        reply = ''
        for i in range(5):
            reply += self.ser.readline().decode('utf-8').strip() + '  '

        if output:
            print('firmware version, serial number, product code, operating time, \
    the number of times the laser diode was turned on.')
            print(reply)
            return
        else:
            return reply

    def get_optime(self, output=False):
        """receive laser operation time
        if output=True, will spill the whole string with info, otherwise only returns it
        """
        _ = self.ser.write(self.cmds["get_optime"].encode())
        # output comes in two lines, combine it to one list
        reply1 = self.ser.readline().decode('utf-8').strip()[:-1]  # remove trailing dot
        reply2 = self.ser.readline().decode('utf-8').strip()
        reply = reply1 + ', ' + reply2
        if output:
            print('operating hours and how many times the laser diode has been turned on')
            print(reply)
            return
        else:
            return reply

    def get_access_level(self):
        """receive access level"""
        _ = self.ser.write(self.cmds["get_access_level"].encode())
        reply = self.ser.readline().decode('utf-8').strip()
        access_level = int(reply.split()[2])
        return access_level

    def set_access_level(self, level, input_code=''):
        """change user access level (default level over serial is 0)
        requires setting the correct codes that need to be obtained from the supplier (level 2-3)
        """
        if level == 0:
            print('cannot return to access level 0')
            return

        if not input_code:
            if level == 1:
                input_code = '1234'  # code supplied in user manual
            elif (level == 2) or (level == 3):
                input_code = input('please input access level code for level ' + str(level) + '\n')
                # check if code is numerical
                try:
                    code = int(input_code)
                except ValueError:
                    print('please input a 5 digit number as access level code')
                    return
            else:
                print('invalid access level, please choose a number in [0..3]')
                return

        cmd = self.cmds["set_access_level"] + ' ' + str(level) + ' ' + input_code
        _ = self.ser.write(cmd.encode())
        reply = self.ser.readline().decode('utf-8').strip()
        # check if access level code worked
        if reply == '<ERR 4>':
            print('invalid code to unlock access level', str(level), ': ', input_code)
        else:
            self.check_reply(reply)
        print('Access level:', str(self.get_access_level()))

    # --------------------------------------------------------------------------
    # modify settings

    def set_crystal_temp(self, settemp):
        """set crystal temp (centi-degC, eg 2550)"""
        if self.get_access_level() < 3:
            print('not enough privilege, please update access level to 3 first')
            return

        currtemp = self.get_crystal_temp()
        # check if there is actually a crystal
        if currtemp < 0:
            print('no crystal, ignoring input')
            return

        old_settemp = self.get_crystal_set_temp()
        if (settemp > 2500) & (settemp < 3500):
            _ = self.ser.write((self.cmds["set_crystal_temp"] + ' ' + str(settemp)).encode())
            reply = self.ser.readline().decode('utf-8').strip()
            self.check_reply(reply)
            new_settemp = self.get_crystal_set_temp()
        else:
            print('please give a reasonable input temperature (2500-3500) as integer')

    def set_diode_temp(self, settemp):
        """set laser diode temp (centi-degC, eg 2550)"""
        if self.get_access_level() < 3:
            print('not enough privilege, please update access level to 3 first')
            return

        currtemp = self.get_diode_temp_num()
        old_settemp = self.get_diode_set_temp()
        if (settemp > 2500) & (settemp < 3500):
            _ = self.ser.write((self.cmds["set_diode_temp"] + ' ' + str(settemp)).encode())
            reply = self.ser.readline().decode('utf-8').strip()
            self.check_reply(reply)
            new_settemp = self.get_diode_set_temp()
        else:
            print('please give a reasonable input temperature (2500-3500) as integer')

    def set_diode_current(self, setcurr):
        """set diode current (mA)"""
        if self.get_access_level() < 3:
            print('not enough privilege, please update access level to 3 first')
            return

        currcurr = self.get_diode_current()
        old_setcurr = self.get_diode_set_current()
        currlimit = self.get_diode_current_limit()
        if setcurr <= currlimit:
            _ = self.ser.write((self.cmds["set_diode_current"] + ' ' + str(setcurr)).encode())
            reply = self.ser.readline().decode('utf-8').strip()
            self.check_reply(reply)
            new_setcurr = self.get_diode_set_current()
        elif setcurr > currlimit:
            print('current larger than diode current limit, please set a current less than', currlimit, 'mA')
        else:
            print('please give a reasonable input current')

    def set_opt_power(self, setpower):
        """set optical power (mW)"""
        if self.get_access_level() < 1:
            print('not enough privilege, please update access level to 1 first')
            return

        oldsetpower = self.get_opt_set_power()
        if oldsetpower == 'nan':
            print('sorry, optical power can not be changed on this device')
        else:
            _ = self.ser.write((self.cmds["set_opt_power"] + ' ' + str(setpower)).encode())
            reply = self.ser.readline().decode('utf-8').strip()
            self.check_reply(reply)
            newsetpower = self.get_opt_set_power()
        return

    def set_DAC_value(self, setvalue):
        """set feedback DAC value (0..8191)"""
        if self.get_access_level() < 1:
            print('not enough privilege, please update access level to 1 first')
            return

        old_setvalue = self.get_DAC_set_value()
        if (setvalue > 0) & (setvalue <= 8191):
            _ = self.ser.write((self.cmds["set_feedback_DAC"] + ' ' + str(setvalue)).encode())
            reply = self.ser.readline().decode('utf-8').strip()
            self.check_reply(reply)
            new_setvalue = self.get_DAC_set_value()
        else:
            print('please give a reasonable input value [0..8191]')

    def set_fan_temp(self, settemp):
        """set fan temp (should be in centi-degC, eg 2550)"""
        if self.get_access_level() < 1:
            print('not enough privilege, please update access level to 1 first')
            return

        old_settemp = self.get_fan_set_temp()
        if (settemp > 2500) & (settemp < 3500):
            _ = self.ser.write((self.cmds["set_fan_temp"] + ' ' + str(settemp)).encode())
            reply = self.ser.readline().decode('utf-8').strip()
            self.check_reply(reply)
            new_settemp = self.get_fan_set_temp()
        else:
            print('please give a reasonable input temperature (2500-3500) as integer')

    def enable_autostart(self):
        """enable autostart on power-on"""
        if self.get_access_level() < 1:
            print('not enough privilege, please update access level to 1 first')
            return

        errorcode = self.ser.write((self.cmds["enable_autostart"] + ' 1').encode())
        reply = self.ser.readline().decode('utf-8').strip()
        self.check_reply(reply)

    def disable_autostart(self):
        """disable autostart on power-on"""
        if self.get_access_level() < 1:
            print('not enough privilege, please update access level to 1 first')
            return

        errorcode = self.ser.write((self.cmds["enable_autostart"] + ' 0').encode())
        reply = self.ser.readline().decode('utf-8').strip()
        self.check_reply(reply)

    def save_changes(self):
        """save changes"""
        if self.get_access_level() < 1:
            print('not enough privilege, please update access level to 1 first')
            return

        errorcode = self.ser.write(self.cmds["save_changes"].encode())
        reply = self.ser.readline().decode('utf-8').strip()
        self.check_reply(reply)

    # --------------------------------------------------------------------------
    # get enable/disable output

    def start_laser(self):
        """enable laser output"""
        _ = self.ser.write((self.cmds["enable_laser"] + ' 1').encode())
        reply = self.ser.readline().decode('utf-8').strip()
        self.check_reply(reply)

    def stop_laser(self):
        """disable laser output"""
        errorcode = self.ser.write((self.cmds["enable_laser"] + ' 0').encode())
        reply = self.ser.readline().decode('utf-8').strip()
        self.check_reply(reply)

    def warmup_laser(self):
        """enable laser warmup (not well-documented. seems to be a setting that needs to be done once)"""
        errorcode = self.ser.write(self.cmds["laser_warmup"].encode())
        reply = self.ser.readline().decode('utf-8').strip()
        self.check_reply(reply)

    # --------------------------------------------------------------------------
    # get single readings

    def get_diode_temp(self):
        """get diode temperature from readings"""
        diode_temp = self.get_readings().split()[1]
        print('laser diode temperature: ', diode_temp, 'degC')

    def get_diode_temp_num(self):
        """get diode temperature from readings only as a number"""
        diode_temp = self.get_readings().split()[1]
        diode_temp_num = float(diode_temp)
        return diode_temp_num

    def get_crystal_temp(self):
        """get laser crystal temperature from readings"""
        crystal_temp = self.get_readings().split()[2]
        if float(crystal_temp) < 0:
            print('system has no crystal')
        else:
            print('crystal temperature: ', crystal_temp, 'degC')

    def get_base_temp(self):
        """get laser base temperature from readings"""
        base_temp = self.get_readings().split()[3]
        print('laser base temperature: ', base_temp, 'degC')

    def get_base_temp_num(self):
        """get laser base temperature from readings only as a number"""
        base_temp = self.get_readings().split()[3]
        base_temp_num = float(base_temp)
        return base_temp_num

    def get_diode_current(self):
        """get laser diode current from readings"""
        diode_current = self.get_readings().split()[4]
        print('laser diode current: ', diode_current)

    def get_diode_current_num(self):
        """get laser diode current from readings only as a number"""
        diode_current = self.get_readings().split()[4]
        diode_current_num = float(diode_current.split('m')[0])
        return diode_current_num

    def get_TEC_load(self):
        """get crystal/diode TEC loads from readings"""
        TEC_load_crystal = self.get_readings().split()[5]
        TEC_load_diode = self.get_readings().split()[6]
        print('crystal TEC load: ', TEC_load_crystal)
        print('  diode TEC load: ', TEC_load_diode)

    def laser_status(self):
        """get laser status (OFF, WRM, APC)"""
        status = self.get_readings().split()[7]
        print('system status:', status)

    def is_off(self):
        """check if laser is off"""
        status = self.get_readings().split()[7]
        if status == "OFF":
            print('laser is off')
            return True
        else:
            print('laser is not off, status:', status)
            return False

    def get_fan_load(self):
        """get fan load from readings"""
        fan_load = self.get_readings().split()[8]
        print('fan load: ', fan_load)

    def get_input_voltage(self):
        """get input voltage from readings"""
        input_voltage = self.get_readings().split()[9]
        print('input voltage: ', input_voltage)

    # --------------------------------------------------------------------------
    # get single settings

    def get_crystal_set_temp(self):
        """get crystal set temperature from settings"""
        crystal_set_temp = float(self.get_settings().split()[1]) / 100
        print('crystal set temperature: ', str(crystal_set_temp), 'degC')

    def get_diode_set_temp(self):
        """get laser diode set temperature from settings"""
        diode_set_temp = float(self.get_settings().split()[2]) / 100
        print('laser diode set temperature: ', str(diode_set_temp), 'degC')

    def get_diode_set_current(self):
        """get laser diode set current from settings"""
        diode_set_current = self.get_settings().split()[3]
        print('laser diode set current: ', diode_set_current, 'mA')

    def get_DAC_set_value(self):
        """get feedback DAC set value from settings"""
        DAC_set_value = self.get_settings().split()[4]
        print('feedback DAC set value: ', DAC_set_value)

    def get_opt_set_power(self):
        """get optical power set current from settings"""
        optical_set_power = self.get_settings().split()[5]
        print('set optical power: ', optical_set_power)

    def get_diode_current_limit(self):
        """get laser diode current limit from settings"""
        diode_current_limit = self.get_settings().split()[6]
        print('laser diode current limit: ', diode_current_limit, 'mA')

    def get_autostart_mode(self):
        """get autostart mode from settings (ON/OFF/WRM)"""
        autostart_mode = self.get_settings().split()[7]
        print('autostart mode: ', autostart_mode)

    def get_fan_set_temp(self):
        """get fan set temperature from settings"""
        fan_set_temp = self.get_settings().split()[9]
        print('fan set temperature: ', str(float(fan_set_temp) / 100), 'degC')

    # --------------------------------------------------------------------------
    # get single information

    def get_firmware_version(self):
        """get firmware version from information"""
        firmware_version = self.get_info().split('  ')[0]
        print(firmware_version)

    def get_serial_number(self):
        """get serial number from information"""
        serial_number = self.get_info().split('  ')[1]
        print(serial_number)

    def get_laser_model(self):
        """get laser model from information"""
        laser_model = self.get_info().split('  ')[2]
        print(laser_model)

    def get_optime2(self):
        """get operating time from information (also has a separate serial command)"""
        optime = self.get_info().split('  ')[3]
        print('operating time: ', optime[:-1])  # remove trailing dot

    def get_switch_times(self):
        """get switch times from information (could also be gotten through optime command)"""
        switch_times = self.get_info().split('  ')[4]
        print('laser diode turned on ', switch_times)
