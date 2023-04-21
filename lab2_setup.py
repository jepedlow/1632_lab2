
# %matplotlib inline
import matplotlib, copy
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import gridspec
from matplotlib.cm import get_cmap

import numpy as np
from numpy.random import randn
from numpy import eye, array, asarray, exp

from math import sqrt
from scipy.linalg import expm
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import sqrtm
from scipy import linalg as la
from scipy.integrate import odeint

import sympy as sym

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets 
from IPython.display import display

# float_formatter = "{:.4f}".format
# np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
# np.set_printoptions(precision=3)
# plt.rcParams["font.serif"] = "cmr12"
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams["figure.dpi"] = 100

# %gui qt
import time, sys

import numpy as np
np.set_printoptions(linewidth=130)

# %matplotlib notebook
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 6]

import ipywidgets as widgets
from IPython.display import display

# from jupyterplot import ProgressPlot # see https://github.com/lvwerra/jupyterplot

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui

import esp32imu

class IMU:
    def __init__(self, Fs=100, port='/dev/ttyUSB0', baud=2000000):
        self.Fs = Fs # sample rate
        self.port = port
        self.baud = baud
        
        self.latestimu = None
        
        self._connect()
    
    def __del__(self):
        self._disconnect()
    
    def _connect(self):
        """
        Connect to the IMU via the esp32imu driver.
        
        A callback is registered that fires each time an IMU sample is received.
        """       
        self.driver = esp32imu.SerialDriver(self.port, self.baud) # start the serial driver to get IMU
        time.sleep(0.1) # requisite 100ms wait period for everything to get setup
        # Request a specific IMU sample rate.
        # the max sample rates of accel and gyro are 4500 Hz and 9000 Hz, respectively.
        # However, the sample rate requested is for the entire device. Thus, if a sample
        # rate of 9000 Hz is requested, every received data packet will have a new gyro
        # sample but repeated accelerometer samples
        self.driver.sendRate(self.Fs) # Hz
        time.sleep(0.1) # requisite 100ms wait period for everything to get setup

        # everytime an IMU msg is received, call the imu_cb function with the data
        self.driver.registerCallbackIMU(self._callback)
        
    def _disconnect(self):
        # make sure to clean up to avoid threading / deadlock errors
        self.driver.unregisterCallbacks()
    
    def _callback(self, msg):
        """
        IMU Callback

        Every new IMU sample received causes this function to be called.

        Parameters
        ----------
        msg : ti.SerialIMUMsg
            has fields: ['t_us', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        """
        # unpack data
        self.latestimu = {
            't': msg.t_us*1e-6,
            'acc': np.array([msg.accel_x, msg.accel_y, msg.accel_z]),
            'gyr': np.array([msg.gyro_x, msg.gyro_y, msg.gyro_z])
        }
        
    def reset(self):
        """
        Resets the IMU connection, which resets time
        """
        self._disconnect()
        self._connect()
        
    def get_time(self):
        """
        Get latest timestamp from IMU
        
        Returns
        -------
        t : float
            time (in seconds) of latest IMU measurement
        """
        return self.latestimu['t'] if 't' in self.latestimu else None
        
    def get_acc(self):
        """
        Get latest acceleration measurement from IMU
        
        Returns
        -------
        acc : (3,) np.array
            x, y, z linear acceleration
        """
        return self.latestimu['acc'] if 'acc' in self.latestimu else None
    
    def get_gyr(self):
        """
        Get latest gyro measurement from IMU
        
        Returns
        -------
        gyr : (3,) np.array
            x, y, z angular velocity
        """
        return self.latestimu['gyr'] if 'gyr' in self.latestimu else None



def get_acc_angles(a):
    """
    Compute tilt angle (roll, pitch) from accelerometer data
    
    Parameters
    ----------
    a : (3,) np.array
        measurement vector from accelerometer
    
    Return
    ------
    y : (2,) np.array
        [ϕ, θ] in radians
    """
    
    ϕ = np.arctan2(a[1], np.sqrt(a[0]**2 + a[2]**2))
    θ = np.arctan2(-a[0], np.sqrt(a[1]**2 + a[2]**2))
    
    return np.array([ϕ, θ])



print("Lab 2 is set up")