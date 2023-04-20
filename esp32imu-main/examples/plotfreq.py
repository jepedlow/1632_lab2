#!/usr/bin/python3
import time, sys
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift

import esp32imu

class IMUAnalyzer:
    # Number of samples used to calculate DFT via FFT
    FFT_SIZE = 1024

    # Window function is used since we are looking at small chunks of signal
    # as they arrive in the buffer. This makes the signal look "locally staionary".
    WINDOW = 'hann'

    # How many seconds of time-domain samples to plot
    SAMPLE_WINDOW_SEC = 5

    # Plotting frequency for time-domain signals
    SAMPLE_PLOT_FREQ_HZ = 20

    def __init__(self, sensor='accel'):

        # Which sensor to analyze
        self.sensor = sensor # 'accel' or 'gyro'

        #
        # FFT setup
        #

        self.sigwin = signal.windows.get_window(self.WINDOW, self.FFT_SIZE)

        # data
        self.Fs = 1
        self.last_t_us = 0
        self.t = []
        self.sensx = []
        self.sensy = []
        self.sensz = []
        self.buf_sensx = []
        self.buf_sensy = []
        self.buf_sensz = []

        self.dts = []
        self.hzs = []

        #
        # Plotting setup
        #

        sens = "Accelerometer" if self.sensor == 'accel' else "Gyro"

        # initialize Qt gui application and window
        self.app = QtWidgets.QApplication([])
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle(f"{sens} Analyzer")
        self.window.resize(1000, 800)
        # self.window.setBackground('k')

        self.tdiffwin = QtWidgets.QWidget()
        self.tdiffwin.setWindowTitle("Sample Rate")
        self.tdiffwin.resize(1000, 800)


        # create plots
        self.pw = pg.PlotWidget(title=sens)
        self.pw2 = pg.PlotWidget(title="Spectrum")
        self.tpw = pg.PlotWidget(title="dt")
        self.tpw2 = pg.PlotWidget(title="Sample Rate")

        # create the layout and add widgets
        self.layout = QtWidgets.QGridLayout()
        self.window.setLayout(self.layout)
        self.layout.addWidget(self.pw, 0, 0)
        self.layout.addWidget(self.pw2, 1, 0)

        # create the layout and add widgets
        self.layout2 = QtWidgets.QGridLayout()
        self.tdiffwin.setLayout(self.layout2)
        self.layout2.addWidget(self.tpw, 0, 0)
        self.layout2.addWidget(self.tpw2, 1, 0)

        self.window.show()
        self.tdiffwin.show()

        #
        # Plotting loop
        #
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._timer_cb)
        self.timer.start(int(1e3/self.SAMPLE_PLOT_FREQ_HZ)) # ms

        #
        # Connect to Teensy
        #

        # initialize serial communications to Teensy
        # self.driver = esp32imu.SerialDriver('/dev/ttyUSB0', 2000000)
        self.driver = esp32imu.UDPDriver()
        time.sleep(0.1) # wait for everything to initialize
        self.driver.sendRate(500)

        msg = esp32imu.RGBLedCmdMsg()
        msg.r = 255
        msg.g = 0
        msg.b = 0
        msg.brightness = 100
        self.driver.sendRGBLedCmd(msg)
        print(msg)

        # Connect an IMU callback that will fire when a sample arrives
        self.driver.registerCallbackIMU(self._imu_cb)

        # Block on application window
        self.app.exec()

        # clean up to prevent error or resource deadlock
        self.driver.unregisterCallbacks()

    def _timer_cb(self):
        self.pw.plot(self.t, self.sensx, pen=(1,3), clear=True)
        self.pw.plot(self.t, self.sensy, pen=(2,3))
        self.pw.plot(self.t, self.sensz, pen=(3,3))

        # always keep x-axis auto range based on SAMPLE_WINDOW_SEC
        self.pw.enableAutoRange(axis=pg.ViewBox.XAxis)


        (f, mag) = self._calcSpectrum(self.buf_sensx); self.pw2.plot(f, mag, pen=(1,3), clear=True)
        (f, mag) = self._calcSpectrum(self.buf_sensy); self.pw2.plot(f, mag, pen=(2,3))
        (f, mag) = self._calcSpectrum(self.buf_sensz); self.pw2.plot(f, mag, pen=(3,3))
        self.pw2.enableAutoRange(axis=pg.ViewBox.XAxis)



        self.tpw.plot(self.t, self.dts, pen=(1,3))
        self.tpw2.plot(self.t, self.hzs, pen=(1,3))

        # "drawnow"
        self.app.processEvents()

    def _imu_cb(self, msg):
        dt = (msg.t_us - self.last_t_us) * 1e-6 # us to s
        self.last_t_us = msg.t_us
        hz = 1./dt
        self.Fs = hz
        print('Got IMU at {} us ({:.0f} Hz): {:.2f}, {:.2f}, {:.2f}, \t {:.2f}, {:.2f}, {:.2f}'
                .format(msg.t_us, hz,
                        msg.accel_x, msg.accel_y, msg.accel_z,
                        msg.gyro_x, msg.gyro_y, msg.gyro_z))

        if self.sensor == 'accel':
            sensx = msg.accel_x
            sensy = msg.accel_y
            sensz = msg.accel_z
        else:
            sensx = msg.gyro_x
            sensy = msg.gyro_y
            sensz = msg.gyro_z

        # FIFO buffer for time-domain plotting
        self.t.append(msg.t_us / 1e6)
        self.sensx.append(sensx)
        self.sensy.append(sensy)
        self.sensz.append(sensz)
        self.dts.append(dt)
        self.hzs.append(hz)

        if len(self.t) > hz*self.SAMPLE_WINDOW_SEC:
            self.t.pop(0)
            self.sensx.pop(0)
            self.sensy.pop(0)
            self.sensz.pop(0)
            self.dts.pop(0)
            self.hzs.pop(0)

        # FIFO buffer for FFT
        self.buf_sensx.append(sensx)
        self.buf_sensy.append(sensy)
        self.buf_sensz.append(sensz)

        if len(self.buf_sensx) > self.FFT_SIZE:
            self.buf_sensx.pop(0)
            self.buf_sensy.pop(0)
            self.buf_sensz.pop(0)


    def _calcSpectrum(self, buf):
        if len(buf) < self.FFT_SIZE:
            return [], []

        # get rid of DC
        buf = np.array(buf) - np.mean(np.array(buf))

        # window the data for a better behaved short-time FT style DFT
        data = np.array(buf) * self.sigwin

        # compute DFT via FFT
        Y = fft(data)

        # make DFT look as you'd expect, plotting real part
        Y = (np.abs(fftshift(Y))/self.FFT_SIZE)

        # compute frequency bins
        f = fftshift(fftfreq(self.FFT_SIZE, 1./self.Fs))

        return f, Y

if __name__ == '__main__':
    sensor = 'accel' # 'accel' or 'gyro'
    analyzer = IMUAnalyzer(sensor)
