import lab2_setup
import numpy as np
import time
import matplotlib.pyplot as plt

class AccKF():
    def __init__(self):
        self.ts = 0.01
        self.n = 4
        self.ntime = 1000

        self.Qc = 1
        self.Rc = np.diag([0.1, 0.1])


        self.Ad = np.array([[1,self.ts,0,0],
                            [0,1,0,0],
                            [0,0,1,self.ts],
                            [0,0,0,1]])
        self.Vc = self.Qc * np.array([[1/3*self.ts**3,1/2*self.ts**2,0,0],
                            [1/2*self.ts**2, self.ts, 0, 0],
                            [0,0,1/3*self.ts**3,1/2*self.ts**2],
                            [0,0,1/2*self.ts**2, self.ts]])
        self.C = np.array([[1,0,0,0],[0,0,1,0]])


        # x and Q + history
        self.Q = self.Vc #np.eye(self.n)*100
        self.x = np.zeros((self.n,1))
        self.xh = np.empty((self.n,1,self.ntime))
        self.Qh = np.empty((self.n,self.n,self.ntime))
        self.i = 0



        self.imu = lab2_setup.IMU()
        time.sleep(1.0)


    def propagate(self):
        # append to history
        # print(self.x)
        self.xh[:,:,self.i] = self.x
        self.Qh[:,:,self.i] = self.Q
        self.i += 1

        # get Lk
        L = self.Q@self.C.T@ np.linalg.inv(self.C@self.Q@self.C.T + self.Rc)

        # update x and Q estimate with y
        y = np.zeros((2,1))
        acc = self.imu.get_acc()
        y[0] = acc[0]
        y[1] = acc[1]
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
        return

acckf = AccKF()
acckf.run()

t = np.linspace(0,acckf.ntime*acckf.ts,acckf.ntime)

plt.figure()
plt.plot(t,acckf.xh[0,0,:])
plt.show()