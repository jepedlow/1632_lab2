import lab2_setup
import numpy as np
import time
import matplotlib.pyplot as plt

class GyroKF():
    def __init__(self):
        self.ts = 0.01
        self.n = 6
        self.ntime = 100

        self.Qc = 1
        self.Rc = np.diag([0.1, 0.1, 0.1])


        self.Ad = np.array([[1,self.ts,0,0,0,0],
                            [0,1,0,0,0,0],
                            [0,0,1,self.ts,0,0],
                            [0,0,0,1,0,0],
                            [0,0,0,0,1,self.ts],
                            [0,0,0,0,0,1]])
        self.Vc = self.Qc * np.array([[1/3*self.ts**3,1/2*self.ts**2,0,0,0,0],
                            [1/2*self.ts**2, self.ts, 0, 0,0,0],
                            [0,0,1/3*self.ts**3,1/2*self.ts**2,0,0],
                            [0,0,1/2*self.ts**2, self.ts,0,0],
                            [0,0,0,0,1/3*self.ts**3,1/2*self.ts**2],
                            [0,0,0,0,1/2*self.ts**2, self.ts]])
        self.C = np.array([[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]])


        # x and Q + history
        self.Q = self.Vc
        self.x = np.zeros((self.n,1))
        self.xh = np.empty((self.n,1,self.ntime))
        self.Qh = np.empty((self.n,self.n,self.ntime))
        self.i = 0



        # self.imu = lab2_setup.IMU()
        # time.sleep(1.0)


    def propagate(self):
        # append to history
        self.xh[:,:,self.i] = self.x
        self.Qh[:,:,self.i] = self.Q
        self.i += 1

        # get Lk
        L = self.Q@self.C.T@ np.linalg.inv(self.C@self.Q@self.C.T + self.Rc)

        # update x and Q estimate with y
        y = np.zeros((3,1))
        # acc = self.imu.get_gyr()
        # y[0] = acc[0]
        # y[1] = acc[1]
        # y[2] = acc[2]
        xkk = self.x + L@(y - self.C@self.x)
        Qkk = (np.eye(self.n) - L@self.C)@self.Q

        # update x and Q
        x_new = self.Ad@xkk
        Q_new = self.Ad@Qkk@self.Ad.T + self.Vc
        self.x = x_new
        self.Q = Q_new

        return

    def run(self):
        while self.i < self.ntime:
            self.propagate()
            time.sleep(self.ts)

gyrokf = GyroKF()
gyrokf.run()

t = np.linspace(0,gyrokf.ntime*gyrokf.ts,gyrokf.ntime)


plt.figure(1)
plt.plot(t,gyrokf.xh[0,0,:], label="$\phi$")
plt.plot(t,gyrokf.xh[2,0,:], label="$\\theta$")
plt.plot(t,gyrokf.xh[4,0,:], label="$\psi$")
plt.xlabel("Time [s]")
plt.title("Gyro Only: State")
plt.ylabel("Orientation [rad]")
plt.legend()
plt.show()



plt.figure(2)
plt.plot(t,gyrokf.Qh[0,0,:], label="$Q_1$")
plt.plot(t,gyrokf.Qh[1,1,:], label="$Q_2$")
plt.plot(t,gyrokf.Qh[2,2,:], label="$Q_3$")
plt.plot(t,gyrokf.Qh[3,3,:], label="$Q_4$")
plt.plot(t,gyrokf.Qh[4,4,:], label="$Q_5$")
plt.plot(t,gyrokf.Qh[5,5,:], label="$Q_6$")
plt.xlabel("Time [s]")
plt.title("Gyro Only: Q")
plt.ylabel("Q")
plt.legend()
plt.show()