import numpy as np
import pandas as pd
from typing import Dict
import model

class Simulation:
    '''
    Classs to run and save simulations.
    '''
    def __init__(self, num_simulations: int, num_CCs: int, num_users: int, alphas, num_steps: int, 
                 random_seed: int = 42, evolution: int = 0, max_follows: int = 0):
        '''
        Initialize the simulation object.
        params:
            num_CCs: number of CCs in the network
            num_users: number of users in the network
            alpha: popularity bias parameter
            num_steps: number of steps to run the simulation for; if num_steps == None, the simulation will run until convergence
            random_seed: random seed for the simulation
        '''
        # set the random seed
        self.gen = np.random.RandomState(random_seed)
        self.num_CCs = num_CCs
        # store params
        self.num_users = num_users
        self.alphas = alphas
        self.num_steps = num_steps
        self.num_simulations = num_simulations
        self.evolution = evolution
        self.max_follows = max_follows

        self.results = []

    def simulate(self) -> Dict[int, dict]:
        '''Runs a simulation, for the parameters in the config file.
        
        returns a dictionary with the results of the simulation
        such that dict[i] returns the result of the i'th simulation
        '''
        
        self.results = {}
        for i in range(self.num_simulations):
            # variables to track during simulation
            data = {}
            num_iterations = 0
            # create the platform
            p = model.Platform(self.num_users, self.num_CCs, self.alphas, self.gen, self.evolution, self.max_follows)

            # iterate the platform either num_steps or until convergence
            if self.num_steps:
                did_converge = False
                for _ in range(self.num_steps):
                    # iterate only if it didn't converge so far
                    if not did_converge:
                        did_converge = p.iterate()
                        num_iterations += 1
                    else:
                        break
            else:
                while not p.check_convergence():
                    num_iterations += 1
                    p.iterate()

            # record statistics after the runs
            data['timesteps'] = num_iterations
            data['num_followers'] = p.network.num_followers
            data['num_followees'] = p.network.num_followees
            data['num_timestep_users_found_best'] = p.users_found_best
            data['average_pos_best_CC'] = p.average_pos_best_CC
            data['did_converge'] = p.check_convergence()
            data['user_satisfaction'] = [u.best_followed_CC.id for u in p.network.users]
            if self.evolution:
                data['evolutionary_data'] = p.evolutionary_data

            # data['G'] = p.network.G.tolist() #instead of hte whole network it woould be good to have aggregates here aoready.
            self.results[i] = data

        return self.results