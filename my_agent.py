__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

from typing import Tuple, List

import numpy as np

agentName = "<my_agent>"
trainingSchedule = [("random_agent.py", 5), ("self", 1)]  # Train against random agent for 5 generations,


# then against self for 1 generation

# This is the class for your cleaner/agent
class Cleaner:

    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        # This is where agent initialisation code goes (including setting up a chromosome with random values)

        # Leave these variables as they are, even if you don't use them in your AgentFunction - they are
        # needed for initialisation of children Cleaners.
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns

    def AgentFunction(self, percepts):

        # The percepts are a tuple consisting of four pieces of information
        #
        # visual - it information of the 3x5 grid of the squares in front and to the side of the cleaner; this variable
        #          is a 3x5x4 tensor, giving four maps with different information
        #          - the dirty,clean squares
        #          - the energy
        #          - the friendly and enemy cleaners that are able to traverse vertically
        #          - the friendly and enemy cleaners that are able to traverse horizontally
        #
        #  energy - int value giving the battery state of the cleaner -- it's effectively the number of actions
        #           the cleaner can still perform before it runs out of charge
        #
        #  bin    - number of free spots in the bin - when 0, there is no more room in the bin - must emtpy
        #
        #  fails - number of consecutive turns that the agent's action failed (rotations always successful, forward or
        #          backward movement might fail if it would result in a collision with another robot); fails=0 means
        #          the last action succeeded.

        visual, energy, bin, fails = percepts

        # You can further break down the visual information

        floor_state = visual[:, :, 0]  # 3x5 map where -1 indicates dirty square, 0 clean one
        energy_locations = visual[:, :, 1]  # 3x5 map where 1 indicates the location of energy station, 0 otherwise
        vertical_bots = visual[:, :,
                        3]  # 3x5 map of bots that can in this turn move up or down (from this bot's point of
        # view), -1 if the bot is an enemy, 1 if it is friendly
        horizontal_bots = visual[:, :,
                          3]  # 3x5 map of bots that can in this turn move up or down (from this bot's point
        # of view), -1 if the bot is an enemy, 1 if it is friendly

        # You may combine floor_state and energy_locations if you'd like: floor_state + energy_locations would give you
        # a mape where -1 indicates dirty square, 0 a clean one, and 1 an energy station.

        # You should implement a model here that translates from 'percepts' to 'actions'
        # through 'self.chromosome'.
        #
        # The 'actions' variable must be returned, and it must be a 4-item list or a 4-dim numpy vector

        # The index of the largest value in the 'actions' vector/list is the action to be taken,
        # with the following interpretation:
        # largest value at index 0 - move forward;
        # largest value at index 1 - turn right;
        # largest value at index 2 - turn left;
        # largest value at index 3 - move backwards;
        #
        # Different 'percepts' values should lead to different 'actions'.  This way the agent
        # reacts differently to different situations.
        #
        # Different 'self.chromosome' should lead to different 'actions'.  This way different
        # agents can exhibit different behaviour.

        # .
        # .
        # .

        # Right now this agent ignores percepts and chooses a random action.  Your agents should not
        # perform random actions - your agents' actions should be deterministic from
        # computation based on self.chromosome and percepts
        action_vector = np.random.randint(low=-100, high=100, size=self.nActions)
        return action_vector

    def create_random_offspring(self, parent1: Tuple[np.ndarray, np.ndarray], parent2: Tuple[np.ndarray, np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates an offspring from two parents using random crossover
        , i.e. combines their two chromosomes into something new


        :param parent1: first parent chromosome
        :param parent2: second parent chromosome
        :return: a new chromosome
        """
        new_weights = np.zeros(self.nActions, self.nPercepts)
        new_biases = np.zeros(self.nActions)
        parent1_weights, parent1_biases = parent1
        parent2_weights, parent2_biases = parent2

        for i in range(len(new_weights)):
            for j in range(len(new_weights[0])):
                rand_num = np.random.rand()
                if rand_num < 0.5:
                    new_weights[i, j] = parent1_weights[i, j]
                else:
                    new_weights[i, j] = parent2_weights[i, j]

        # TODO to create a mutation - need someway to create number from a RANGE of values - challenge is to decide
        # TODO cont: what this range should be. Should weight values be normalised? - This sounds complicated

        for i in range(len(new_biases)):
            rand_num = np.random.rand()
            if rand_num < 0.5:
                new_biases[i] = parent1_biases[i]
            else:
                new_biases[i] = parent2_biases[i]

        return new_weights, new_biases

    def create_random_chromosome(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a random chromosome

        :return: a tuple containing a random 4*63 weights matrix and a ndarray of length 4 containing biases
        """
        weights = np.random.rand(self.nActions, self.nPercepts)
        biases = np.random.rand(self.nActions)
        return weights, biases

    def create_initial_chromosomes(self, population_size) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Creates the initial random population of chromosomes

        :param population_size: size of the initial population to create
        :return: a list of chromosomes
        """
        initial_population = [None] * population_size
        for i in range(len(initial_population)):
            initial_population[i] = self.create_random_chromosome()
        return initial_population

    def create_actions(self, percepts, chromosome) -> List[float]:
        """
        Receives a chromosome and creates the actions that it should take based on the percepts of the game

        Does this in a linear manner

        TODO explore other ways to do this! - i.e. not just in a linear fashion

        :param percepts: 63 length array containing percepts
        :param chromosome: Tuple containing a 4*64 numpy matrix of weights and a ndarray of length 4 containing the biases
        :return:
        """
        results = [0] * self.nActions
        weights, biases = chromosome

        for i in range(len(weights)):
            row = weights[i]
            this_result = 0
            for j in range(len(row)):
                this_result += percepts[j] * row[j]
            this_result += biases[i]
            results[i] = this_result
        return results


def evalFitness(population):
    N = len(population)

    # Fitness initialiser for all agents
    fitness = np.zeros((N))

    # This loop iterates over your agents in the old population - the purpose of this boilerplate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, cleaner in enumerate(population):
        # cleaner is an instance of the Cleaner class that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, each object have 'game_stats' attribute provided by the
        # game engine, which is a dictionary with the following information on the performance of the cleaner in
        # the last game:
        #
        #  cleaner.game_stats['cleaned'] - int, total number of dirt loads picked up
        #  cleaner.game_stats['emptied'] - int, total number of dirt loads emptied at a charge station
        #  cleaner.game_stats['active_turns'] - int, total number of turns the bot was active (non-zero energy)
        #  cleaner.game_stats['successful_actions'] - int, total number of successful actions performed during active
        #                                                  turns
        #  cleaner.game_stats['recharge_count'] - int, number of turns spent at a charging station
        #  cleaner.game_stats['recharge_energy'] - int, total energy gained from the charging station
        #  cleaner.game_stats['visits'] - int, total number of squares visited (visiting the same square twice counts
        #                                      as one visit)

        # This fitness functions considers total number of cleaned squares.  This may NOT be the best fitness function.
        # You SHOULD consider augmenting it with information from other stats as well.  You DON'T HAVE TO make use
        # of every stat.

        # TODO update me!, But for now, should do just for early stage of the project
        fitness[n] = cleaner.game_stats['cleaned']

    return fitness


def newGeneration(old_population):
    # This function should return a tuple consisting of:
    # - a list of the new_population of cleaners that is of the same length as the old_population,
    # - the average fitness of the old population

    N = len(old_population)

    # Fetch the game parameters stored in each agent (we will need them to
    # create a new child agent)
    gridSize = old_population[0].gridSize
    nPercepts = old_population[0].nPercepts
    nActions = old_population[0].nActions
    maxTurns = old_population[0].maxTurns

    fitness = evalFitness(old_population)

    # At this point you should sort the old_population cleaners according to fitness, setting it up for parent
    # selection.
    # .
    # .
    # .

    # Create new population list...
    new_population = list()
    for n in range(N):
        # Create a new cleaner
        new_cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)

        # Here you should modify the new cleaner' chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_cleaner.chromosome

        # Consider implementing elitism, mutation and various other
        # strategies for producing a new creature.

        # .
        # .
        # .

        # Add the new cleaner to the new population
        new_population.append(new_cleaner)

    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    return (new_population, avg_fitness)
