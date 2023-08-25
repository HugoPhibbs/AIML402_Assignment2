__author__ = "Hugo Phibbs"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "phihu4141@student.otago.ac.nz"

from typing import NewType

from typing import Tuple, List

import numpy as np

agentName = "<my_agent>"
trainingSchedule = [("random_agent.py", 5), ("self", 1)]  # Train against random agent for 5 generations,


class Chromosome:
    """
    Custom class for a Chromosome, extracted into a class for easy reading
    
    I decided to keep them separate because I wanted to do cross over of them independently
    Whether this is necessary, I'm not sure, but I'll do it anyways for clarity.
    
    Contains:
    - Weight matrix, dimensions self.nActions * self.nPercepts
    - Bias row vector, dimensions self.nActions * 1
    """

    def __init__(self, weights, biases):
        """
        
        :param weights: self.nActions * self.nPercepts weight matrix
        :param biases:  self.nActions * 1 bias vector
        """
        self.weights = weights
        self.biases = biases


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

    def create_k_point_weights(self, weights_1: np.ndarray, weights_2: np.ndarray, k_points=4) -> np.ndarray:
        """
        Creates a weight offspring from two parents using a k-point crossover

        Does this by crossing over segments of each row, row by row.

        :param parent1: first parent's weight matrix
        :param parent2: second parent's weight matrix
        :return: a new chromosome
        """
        select_parent1 = np.random.choice([True, False])  # Specify whether to start selection from parent1 or parent2
        new_weights = np.zeros_like(weights_1)

        segment_size = len(new_weights) / k_points
        for i in range(len(weights_1)):
            j = 0
            while j < len(weights_1[0]):
                weights_to_select_from = weights_1 if select_parent1 else weights_2

                # Possibly goes beyond end of array, np handles gracefully though
                new_weights[i, j:j + segment_size] = weights_to_select_from[i, j:j + segment_size]

                select_parent1 = not select_parent1
                j += segment_size

        return new_weights

    def create_lin_interpolation_biases(self, biases_1: np.ndarray, biases_2: np.ndarray):
        """
        Performs a random linear interpolation to combine two weights values.

        To keep bias values somewhat more stable than weights, this technique can be used

        :param biases_1: first vector of biases
        :param biases_2: second vector of biases
        :return:
        """
        num_biases = len(biases_1)
        lin_interpolation_vec = np.random.uniform(low=0.4, high=0.6, size=num_biases)
        ones_vec = np.ones(size=num_biases)
        return biases_1 * lin_interpolation_vec + biases_2 * (ones_vec - lin_interpolation_vec)

    def mutate_weights(self, weights: np.ndarray, mutation_rate=0.05):
        """
        Mutates the weights of the chromosomes.

        Mutates with a given rate, and chooses a value between the min_weight - max_min_dist / 4 and max_weight + max_min_dist / 4. This is
        so weight values not yet encountered have the chance to be selected.

        :param weights: weights matrix to mutate
        :param mutation_rate: rate of mutation
        :return:
        """
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        max_min_dist = (max_weight + min_weight) / 2

        for row in weights:
            rand_num = np.random.rand()

            if rand_num < mutation_rate:
                rand_index = np.random.randint(0, len(row))
                row[rand_index] = np.random.uniform(low=min_weight - max_min_dist / 4,
                                                    high=max_weight + max_min_dist / 4)

        return weights

    def create_runif_weights(self, weights_1: np.ndarray, weights_2: np.ndarray) -> np.ndarray:
        """
        Creates a weights matrix from two parent's weights using random uniform crossover

        :param weights_1: weight matrix belonging to the first parent
        :param weights_2: weight matrix belonging to the second parent
        :return: a new weights matrix
        """
        new_weights = np.zeros(self.nActions, self.nPercepts)

        for i in range(len(new_weights)):
            for j in range(len(new_weights[0])):
                rand_num = np.random.rand()
                if rand_num < 0.5:
                    new_weights[i, j] = weights_1[i, j]
                else:
                    new_weights[i, j] = weights_2[i, j]

        # TODO to create a mutation - need someway to create number from a RANGE of values - challenge is to decide
        # TODO cont: what this range should be. Should weight values be normalised? - This sounds complicated
        return new_weights

    def create_random_chromosome(self) -> Chromosome:
        """
        Creates a random chromosome

        :return: a new Chromosome
        """
        weights = np.random.rand(self.nActions, self.nPercepts)
        biases = np.random.rand(self.nActions)
        return Chromosome(weights, biases)

    def tournament_selection(self, population: List[Chromosome], sample_size_factor = 0.1) -> Tuple[Chromosome, Chromosome]:
        """
        Does tournament selection to choose new parents

        :param population: population of Chromosomes to choose parents from
        :param sample_size_factor: size of the subset to create as a ratio of the total population size
        :return: a tuple containing two Chromosomes for the selected parents
        """
        sample_size = int(len(population) * sample_size_factor)
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_fitnesses = np.zeros_like(sample)
        for i in range(len(sample)):
            sample_fitnesses = evalFitness(sample[i])

        top_two_parent_indices = np.argpartition(sample_fitnesses, -2)[-2:]

        return sample_fitnesses[top_two_parent_indices]

    def create_initial_chromosomes(self, population_size: int) -> List[Chromosome]:
        """
        Creates the initial random population of chromosomes

        :param population_size: size of the initial population to create
        :return: a list of chromosomes
        """
        return [self.create_random_chromosome() for i in range(population_size)]

    def create_actions(self, percepts: np.ndarray, chromosome: Chromosome) -> np.ndarray:
        """
        Receives a chromosome and creates the actions that it should take based on the percepts of the game

        Uses a linear model of weights * percepts + biases(Transpose)

        :param percepts: 63 length array containing percepts
        :param chromosome: A Chromosome to create actions for.
        :return:
        """
        return np.matmul(chromosome.weights, percepts) + chromosome.biases.T


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
