import numpy as np
import os
from . import simulators

        
# Defining a class to perform the string method. So far this is only for energy surfaces, not including free energy.
class String:
# The constructor has two options, if the path variable is given, it can load a saved trajectory, if not other variables have to be given to create a new one.
    def __init__(self, potential, number_nodes = 1, gamma = 1, T = 273, minima = 0, force_constants = 40, path = 0):
    ## First exploring the load option if the path variable is given.
        if isinstance(path, str):
            self.trajectory = []                #This is the trajectory of the node positions, the trajectory of the simulations are saved in the respective objects.
### Loading the node positions into trajectory for as many nodes as the files are available
            i = 0
            while True:
                try:
                    self.trajectory.append(np.loadtxt(path+"/"+str(i), delimiter=','))
                    i+=1
                except:
                    break
### Only the former part loads saved data, from here on all other parameters are defined as given in the call of the constructor. (Default if none are used.)
            self.force_constants = force_constants
            self.minima_set = True
            if minima == 0:
                self.minima_set = False
            self.number_nodes = len(self.trajectory[0])
            self.nodes = []
### Nodes are initialised with the restraints using the last position in the loaded file.
            for i,node in enumerate(self.trajectory[-1]):
                self.nodes.append(simulators.Langevin(potential,node, gamma, T))
                self.nodes[i].set_restraint(node, force_constants)
    ## Second, if no path is given, a new string is created.
        else:
            self.number_nodes = number_nodes
            self.nodes = []
            self.trajectory = []
            self.force_constants = force_constants
            self.minima_set = True
            if minima == 0:
                self.minima_set = False
            for i in range(number_nodes):
                self.nodes.append(simulators.Langevin(potential, [0,0], gamma, T))
            if self.minima_set:
                self.nodes[ 0].set_current_position(minima[0])
                self.nodes[-1].set_current_position(minima[1])
        

# Method to calculate the distance between two nodes, variables first and second should be node numbers. The type variable is used to specify whether to calculate the distance between the current position of the simulation or the current position of the node (restraint).
    def distance_between_nodes(self, first, second, type = "current_position"):
        # Returns the vector from first to second
        if isinstance(first, int) and isinstance(second, int):
            pass
        else:
            raise Exception("Arguments must be integers decoding the position of the Nodes in the string.")
            return False
        position_keywords = ["current position", "current_position", "position"]
        restraint_keywords = ["current restraint position", "current_restraint_position", "restraint position", "restraint_position", "restraint"]
        if type in position_keywords:
            distance_vector = np.array(self.nodes[second].current_position) - np.array(self.nodes[first].current_position)
            return (np.dot(distance_vector,distance_vector))**0.5, distance_vector
        elif type in restraint_keywords:
            distance_vector = np.array(self.nodes[second].restraint_position) - np.array(self.nodes[first].restraint_position)
            return (np.dot(distance_vector,distance_vector))**0.5, distance_vector
        else:
            print("The specification of the type is not known, please provide any of the following keywords:\ncurrent position, current_position or position\nfor the current position and\ncurrent restraint position, current_restraint_position, restraint position, restraint_position or restraint\nfor the node/restraint position.")
            raise Exception("Please try again with one of the keywords.")
            return False

# Method for preparation, performs minimization of the end nodes if not set already and sets the position of the nodes in between in a linear interpolation       
    def prepare(self, dt = 0.1, steps = 1000, threshold = 0.000001):
        if self.minima_set:
            self.nodes[ 0].minimize(dt*0.1, threshold)
            self.nodes[-1].minimize(dt*0.1, threshold)
        else:
            while self.distance_between_nodes(0,-1)[0] < 1:
                self.nodes[0].simulate(dt, steps)
                self.nodes[-1].simulate(dt, steps)
                self.nodes[ 0].minimize(dt*0.1, threshold)
                self.nodes[-1].minimize(dt*0.1, threshold)
                steps *= 2
            self.minima_set = True
            
        L, distance_vector = self.distance_between_nodes(0,-1)
        restraint_positions = []
        for i in range(self.number_nodes):
            node_position = self.nodes[0].current_position + i/(self.number_nodes-1)*distance_vector
            self.nodes[i].set_current_position(np.array(node_position))
            self.nodes[i].set_restraint(np.array(node_position),self.force_constants)
            restraint_positions.append(node_position)
        self.trajectory.append(np.array(restraint_positions))
        print(f"Preparation done\n Starting node: {self.nodes[0].current_position}\n Ending node:   {self.nodes[-1].current_position}")
        return np.array([self.nodes[0].current_position,self.nodes[0].current_position])

# Method enforces equidistance during the run of the string
    def reposition_nodes(self):
    ## 1. Calculating current alphas (positions on the string)
        current_alphas = [0]
        for i in range(self.number_nodes-1):
            current_alphas.append(current_alphas[-1] + self.distance_between_nodes(i,i+1,"restraint")[0])
        # 2. Asign new equidistant positions by linear interpolation
        new_positions = []
        for i, current_alpha in enumerate(current_alphas):
            # Keep start and end fixed
            if i==0 or i==self.number_nodes-1:
                new_positions.append(self.trajectory[-1][i])
            # Reposition middle nodes
            else:
                new_alpha = i/(self.number_nodes-1)*current_alphas[-1]
                if current_alpha >= new_alpha:
                    distance, vector = self.distance_between_nodes(i-1,i,"restraint")
                    normed_vector = vector/distance
                    new_node_position = self.trajectory[-1][i-1] + (new_alpha-current_alphas[i-1])*normed_vector
                elif current_alpha < new_alpha:
                    distance, vector = self.distance_between_nodes(i,i+1,"restraint")
                    normed_vector = vector/distance
                    new_node_position = self.trajectory[-1][i] - (current_alphas[i]-new_alpha)*normed_vector
                self.nodes[i].set_restraint(np.array(new_node_position), self.force_constants)
                new_positions.append(new_node_position)
        self.trajectory.append(np.array(new_positions))

# Method that performs the string calculation, damping variable refers to the damping in updating the restraint, reposition period defines the number of Langevin steps between running the "reposition_nodes" method.
    def run(self, dt = 0.05, steps = 5_000, reposition_period = 100, damping = 30):
        for step in range(int(steps/reposition_period)):
            print(f"Running step {(step+1)*reposition_period} of {steps}.", end = "\r")
            string_position = [self.nodes[0].restraint_position]
            for i in range(self.number_nodes-2):
                restraint_position = self.nodes[i+1].simulate(dt,reposition_period, damping=damping)
                string_position.append(np.array(restraint_position))
            string_position.append(self.nodes[-1].restraint_position)

            self.trajectory.append(np.array(string_position))
            self.reposition_nodes()

# Method to save the trajectories of the nodes (restraints only) in the format used to load a new string
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for i,traj_point in enumerate(self.trajectory):
            np.savetxt(path+"/"+str(i), traj_point, delimiter=',', fmt='%f')
