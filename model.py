import numpy as np
from typing import List
from collections import defaultdict 


class CC:
    '''
    Class modelling the behavior of con∏tent creators.
    '''
    def __init__(self, id):
        self.id = id


class User:
    '''
    Class modelling the behavior of users.
    '''
    def __init__(self, id):
        self.id = id

        # the best CC followed so far
        self.best_followed_CC = None

    def decide_follow(self, c: CC):
        '''Evaluates whether the user wants to follow CC c.

        input: c - a content creator
        ------
        output: bool - decision if it follows c'''

        # it follows c iff they are better (closer to top) then the best followed so far
        if (self.best_followed_CC is None) or (self.best_followed_CC.id > c.id):
            return True

        return False


class Network:
    '''
    Class capturing a follower network between from users to items, stored as an adjacency matrix.
    '''
    def __init__(self, u_ids: List[int], c_ids: List[int], max_follows: int = 0):
        self.users = [User(u_id) for u_id in u_ids]
        self.CCs = [CC(c_id) for c_id in c_ids]
        # keep track of the number of followers each CC has
        self.num_followers = np.zeros(len(c_ids))
        # keep track of the number of CCs each user follows
        self.num_followees = np.zeros(len(u_ids))
        # adjacency list of the graph
        self.adjacency_list = [[] for _ in range(len(self.users))]

        self.max_follows = np.inf if not max_follows else max_follows

    def follow(self, u_id: int, c_id: int):
        '''
        User u follows content creator c; and updates the Network

        input: u - user
               c - CC
               num_timestep - the iteration number of the platform (int)
               when_users_found_best - a list of length the number of users who keeps the timesteps when each of the user found their best CC (or -1 if they didn't yet)
        '''
        u = self.users[u_id]
        c = self.CCs[c_id]
        if not self.is_following(u, c):
            if u.decide_follow(c) and self.num_followees[u.id] <= self.max_follows:
                self.adjacency_list[u.id].append(c_id)
                self.num_followers[c.id] += 1
                self.num_followees[u.id] += 1
                u.best_followed_CC = c

    def is_following(self, u: User, c: CC) -> bool:
        '''
        Returns True if user u follows content creator c.
        '''
        return c.id in self.adjacency_list[u.id]


class Platform:
    '''
    Class for simulating the recommendation procedure given the parameters
    specified in __init__.
    '''
    def __init__(self, num_users: int, num_CCs: int, alphas: List[float], gen, evolution = 0, max_follows = 0):
        # store params
        self.num_users = num_users
        self.num_CCs = num_CCs
        self.alphas = alphas
        self.gen = gen
        self.num_alphas = len(self.alphas)

        # set up users, CCs and network classes
        self.u_ids = list(range(num_users))
        self.c_ids = list(range(num_CCs))
        self.network = Network(self.u_ids, self.c_ids, max_follows)
        
        # keep track of the timesteps when users found their best CC
        self.users_found_best = [-1]*num_users
        # keep track of the average quality experienced by users
        self.average_pos_best_CC = []
        # list containing the indices of users who have not not converged yet
        self.id_searching_users = list(range(num_users))
        # the platform keeps track of the number of timesteps it has been iterated
        self.timestep = 0

        # keep track of how the network evolves
        self.evolution = evolution
        self.evolutionary_data = dict()

    def iterate(self) -> bool:
        '''
        Makes one iteration of the platform.
        Used only to update the state of the platform
        Returns True if converged else False
        '''

        # 0) the platform starts the next iteration
        self.timestep += 1

        # 1) each user gets a recommendation
        recs = self.recommend()

        # 2) each user decides whether or not to follow the recommended CC
        for u_id in self.id_searching_users: # only the users who have not found the best CC yet (CC_0)
            c_id = recs[u_id]
            self.network.follow(u_id, c_id)
            if c_id == 0: # in this case we follow the best CC to the procedure is done for user u
                self.users_found_best[u_id] = self.timestep
            
        self.update_searching_users()

        if self.evolution and self.timestep % (self.evolution) == 0:
            self.evolutionary_data[self.timestep]['num_followers'] = np.copy(self.network.num_followers)
            self.evolutionary_data[self.timestep]['num_followees'] = np.copy(self.network.num_followees)
            self.evolutionary_data[self.timestep]['user_satisfactions'] = [u.best_followed_CC.id for u in self.network.users]

        return self.check_convergence()

    def update_searching_users(self):
        '''
        Updates the list of users who are still searching for CC_0.
        '''
        self.id_searching_users = list(filter(lambda i: self.users_found_best[i] == -1, self.id_searching_users))

    def recommend(self) -> np.array:
        '''
        input: content_creators - a list of content creators
        num_followers - number of followers of each content creator
        -----
        output: array of CCs chosen based on PA
        '''
        probs = np.zeros(self.num_CCs)
        for alpha in self.alphas:
            prob_choice = np.power(self.network.num_followers + np.ones(self.num_CCs), alpha)
            prob_choice /= sum(prob_choice)
            probs += prob_choice / self.num_alphas
            if self.evolution and self.timestep % (self.evolution) == 0:
                self.evolutionary_data[self.timestep] = {'probs': probs}

        return self.gen.choice(self.c_ids, self.num_users, p=probs)

    def check_convergence(self) -> bool:
        # the platform converged if there are no more searching users (users who can find better CCs)
        return len(self.id_searching_users) == 0