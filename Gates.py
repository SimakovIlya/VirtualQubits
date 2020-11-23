import numpy as np




def fSim_gate(theta, phi):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(theta), -1j*np.sin(theta), 0],
                     [0, -1j*np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, np.exp(-1j*phi)]])