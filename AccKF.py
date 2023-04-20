import lab2_setup
import numpy as np
import time

class AccKF():
    def __init__(self):
        self.ts = 0.01
        self.n = 4

        self.Qc = 1
        self.Rc = np.diag([0.1, 0.1])


        self.Ad = np.array([[1,self.ts,0,0],
                            [0,1,0,0],
                            [0,0,1,self.ts],
                            [0,0,0,1]])
        self.Vd = self.Qc * np.array([[1/3*self.ts**3,1/2*self.ts**2,0,0],
                            [1/2*self.ts**2, self.ts, 0, 0],
                            [0,0,1/3*self.ts**3,1/2*self.ts**2],
                            [0,0,1/2*self.ts**2, self.ts]])
        self.C = np.array([[1,0,0,0],[0,0,1,0]])


        # x and Q + history
        self.Q = np.eye(self.n)*100
        self.x = np.zeros(self.n)
        self.xh = np.empty((self.n,1,))
        self.Qh = np.empty((self.n,self.n,))



        self.imu = lab2_setup.IMU()


    def propagate(self):
        # append to history
        self.xh.append(self.x)
        self.Qh.append(self.Q)

        # get Lk
        L = self.Q@self.C.T@ np.linalg.inv(self.C@self.Q@self.C.T + self.Rc)

        # update x and Q estimate with y
        y = self.imu.get_acc()
        xkk = self.x + L@(y - self.C@self.x)
        Qkk = (np.eye(self.n) - L@self.C)@self.Q

        # update x and Q
        x_new = self.Ad@xkk
        Q_new = self.Ad@Qkk@self.Ad.T + self.Vc
        self.x = x_new
        self.Q = Q_new

        return

    def run(self):
        while True:
            self.propagate()
            time.sleep(self.ts)

acckf = AccKF()
acckf.run()

        