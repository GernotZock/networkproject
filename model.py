import numpy as np

class User:
    '''
    Class modelling the behavior of users.
    '''
    def __init__(self, id):
        self.id = id

        # the best CC followed so far
        self.best_followed_CC = None

    def decide_follow(self, c):
        '''Evaluates whether the user wants to follow CC c.

        input: c - a content creator
        ------
        output: bool - decision if it follows c'''

        # it follows c iff they are better (closer to top) then the best followed so far
        if (self.best_followed_CC is None) or (self.best_followed_CC.id > c.id):
            self.best_followed_CC = c
            return True

        return False


class CC:
    '''
    Class modelling the behavior of content creators.
    '''
    def __init__(self, id):
        self.id = id


class Network:
    '''
    Class capturing a follower network between from users to items, stored as an adjacency matrix.
    '''
    def __init__(self, num_users, num_CCs, G=None, favorite=None):

        self.G = G
        if self.G is None:
            self.G = np.zeros((num_users, num_CCs), dtype=bool)

        self.num_followers = np.count_nonzero(self.G, axis=0)
        self.num_followees = np.count_nonzero(self.G, axis=1)

    def follow(self, u, c, num_timestep, when_users_found_best):
        '''
        User u follows content creator c; and updates the Network

        input: u - user
               c - CC
               num_timestep - the iteration number of the platform (int)
               when_users_found_best - a list of length the number of users who keeps the timesteps when each of the user found their best CC (or -1 if they didn't yet)
        '''

        if not self.G[u.id][c.id]:
            if u.decide_follow(c):
                self.G[u.id][c.id] = True
                self.num_followers[c.id] += 1
                self.num_followees[u.id] += 1

                # if c is the top CC, then u found their best CC this round
                if c.id == 0:
                    when_users_found_best[u.id] = num_timestep

    def is_following(self, u, c):
        '''
        Returns True if user u follows content creator c.
        '''
        return self.G[u.id][c.id]


class Platform:
    '''
    Class for simulating the recommendation procedure given the parameters
    specified in __init__.
    '''
    def __init__(self, num_users, num_CCs, alpha, num_steps, gen):
        # store params
        self.num_users = num_users
        self.num_CCs = num_CCs
        self.alpha = alpha
        self.num_steps = num_steps
        self.gen = gen

        # set up network, users and CCs objects
        self.network = Network(num_users, num_CCs)
        self.users = [User(i)
                      for i in range(num_users)]
        self.CCs = [CC(i)
                    for i in range(num_CCs)]
        
        # keep track of the timesteps when users found their best CC
        self.users_found_best = [-1 for u in self.users]

        # keep track of the average quality experienced by users
        self.average_pos_best_CC = []

        # the users who have not not converged yet
        self.id_searching_users = list(range(num_users))
        
        # the platform keeps track of the number of timesteps it has been iterated
        self.timestep = 0

    def iterate(self):
        '''
        Makes one iteration of the platform.
        Used only to update the state of the platform
        '''

        # 0) the platform starts the next iteration
        self.timestep += 1

        # 1) each user gets a recommendation
        recs = self.recommend()

        # 2) each user decides whether or not to follow the recommended CC
        for u in self.id_searching_users: # only the users who have not found the best CC yet (CC_0)
            self.network.follow(
                self.users[u], recs[u], self.timestep, self.users_found_best)

        self.update_searching_users()
    
    def update_searching_users(self):
        '''
        Updates the list of users who are still searching for CC_0.
        '''
        self.id_searching_users = list(
            filter(lambda i: self.users[i].best_followed_CC.id != 0, self.id_searching_users))

    def recommend(self):
        '''
        input: content_creators - a list of content creators
        num_followers - number of followers of each content creator
        -----
        output: a CC chosen based on PA
        '''


        prob_choice = np.power(self.network.num_followers + np.ones(self.num_CCs), self.alpha)
        prob_choice /= sum(prob_choice)
        return self.gen.choice(self.CCs, self.num_users, p=prob_choice)

    def check_convergence(self):
        # the platform converged if there are no more searching users (users who can find better CCs)
        return len(self.id_searching_users) == 0