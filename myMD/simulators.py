import numpy as np
import scipy.constants as const
from scipy import optimize


class Langevin:
# Class for a Langevin simulator
    def __init__(self, potential, initial, gamma, T):
        self.dimension = len(initial)
        self.gamma = gamma
        self.beta = T*const.k*const.Avogadro/4184
        self.current_position = initial
        self.current_velocity = np.random.normal(0,(2*gamma/self.beta)**0.5)
        self.trajectory = [np.array(initial)]
        self.potential = potential
        self.restraint_position = initial
        self.restraint_force_constant = 0

    def potential(self, r):
        pass

# The gradient of the energy surface is approximated with the scipy method.
    def gradient(self, r):
        return optimize.approx_fprime(r, self.potential)

# Minimize the state of the system with steepest gradient descent.
    def minimize(self, dt, threshold):
        print("Starting minimization from ")
        step = self.gradient(self.current_position)
        print(step)
        while np.dot(np.array(step),np.array(step)) > threshold:
            self.current_position -= step*dt**2
            step = self.gradient(self.current_position)
            print(f"Current gradient is {step}.", end = "\r")
        print(f"Finished minimization with a gradient of {step}.")

# Simulation runs by numerical integration of stochastic integrals, no interpretation is needed as the stochastic force is independent of the position.
    def simulate(self, dt, steps, damping = 0):
    # The damping parameter here is not the damping constant of the Langevin equation, it refers to the damping parameter in the "On the fly string method". For normal Langevin simulations just leave at 0.
        for t in range(steps):
            self.current_velocity += dt*(-self.gradient(self.current_position) 
                                         -self.restraint_force_constant* (np.array(self.current_position) - np.array(self.restraint_position)) 
                                         -self.gamma*self.current_velocity
                                         +np.random.normal(0,(2*self.gamma/self.beta)**0.5))
            self.current_position += dt*self.current_velocity
            self.trajectory.append(np.array(self.current_position))
    #Update the restraint for On the fly string method
            if damping != 0:
                self.restraint_position += (self.current_position - self.restraint_position)/damping
        if damping != 0:
            return self.restraint_position

# Just a method to change the position of the restraint, can also be achieved by directly accessing the variable.
    def set_restraint(self, position, force_constant = 40):
        self.restraint_position = position
        self.restraint_force_constant = force_constant

# Just a method to change the current position of the system, can also be achieved by directly accessing the variable.
    def set_current_position(self, coords):
        self.current_position = coords
