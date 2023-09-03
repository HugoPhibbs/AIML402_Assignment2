__author__ = "Hugo Phibbs"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "phihu4141@student.otago.ac.nz"

from typing import NewType

from typing import Tuple, List

import numpy
import numpy as np

agentName = "Noo-Noo"  # https://teletubbies.fandom.com/wiki/Noo-Noo

num_generations = 25
trainingSchedule = [("random_agent.py", num_generations), ("self", 0)]


class Chromosome:
    """
    Custom class for a Chromosome, extracted into a class for easy reading
    
    I decided to keep them separate because I wanted to do cross over of them independently
    Whether this is necessary, I'm not sure, but I'll do it anyways for clarity.
    
    Contains:
    - Weight matrix, dimensions self.n_actions * self.n_percepts
    - Bias row vector, dimensions self.n_actions * 1
    """

    def __init__(self, weights, biases):
        """
        
        :param weights: self.n_actions * self.n_percepts weight matrix
        :param biases:  self.n_actions * 1 bias vector
        """
        self.weights = weights
        self.biases = biases


class Cleaner:

    def __init__(self, nPercepts: int, nActions: int, gridSize: Tuple[int, int], maxTurns: int,
                 chromosome: Chromosome = None):
        """
        Creates a new cleaner

        :param nPercepts: total number of percepts in this game
        :param nActions: total number of actions that a cleaner can take at any point
        :param gridSize: tuple for the size of the grid (i.e. height * width)
        :param maxTurns: int for the max number of turns that a cleaner can do
        :param chromosome: Chromosome to set to this cleaner, if None, a random one will be set
        """
        self.n_percepts = nPercepts
        self.n_actions = nActions
        self.grid_size = gridSize
        self.max_turns = maxTurns
        self.game_stats = None  # So I don't get annoying IDE warnings, I'm assuming this is set later by the engine

        if chromosome is None:
            self.chromosome = self.create_random_chromosome()
        else:
            self.chromosome = chromosome

    def create_random_chromosome(self) -> Chromosome:
        """
        Creates a random chromosome

        :return: a new Chromosome
        """
        weights = np.random.rand(self.n_actions, self.n_percepts)
        biases = np.random.uniform(low=0, high=self.n_percepts, size=self.n_actions)
        return Chromosome(weights, biases)

    def AgentFunction(self, percepts):
        """
        Returns the actions that this Cleaner should take given a matrix of percepts

        The percepts are a tuple consisting of four pieces of information:

        visual - it information of the 3x5 grid of the squares in front and to the side of the cleaner; this variable
                 is a 3x5x4 tensor, giving four maps with different information
                 - the dirty,clean squares
                 - the energy
                 - the friendly and enemy cleaners that are able to traverse vertically
                 - the friendly and enemy cleaners that are able to traverse horizontally

        energy - int value giving the battery state of the cleaner -- it's effectively the number of actions
              the cleaner can still perform before it runs out of charge
        
        bin_spots- number of free spots in the bin - when 0, there is no more room in the bin - must emtpy
        
        fails - number of consecutive turns that the agent's action failed (rotations always successful, forward or
             backward movement might fail if it would result in a collision with another robot); fails=0 means
             the last action succeeded.

        :param percepts: tuple described as above
        :return: actions vector as described
        """
        visual, energy, bin_spots, num_fails = percepts

        energy_bin_fails = np.array([energy, bin_spots, num_fails])

        # Flatten percepts into an array
        percepts_flattened = np.concatenate((visual.flatten(), energy_bin_fails))

        # Extract energy, bin_spots and fails from the weights, just experimenting to see what happens if I increase their contribution to actions
        energy_bin_fails_weights = self.chromosome.weights[:, -3:]

        # Now calculate actions to take using formula weights * percepts + biases
        return np.matmul(self.chromosome.weights, percepts_flattened) + np.matmul(energy_bin_fails_weights ** 2,
                                                                                  energy_bin_fails) + self.chromosome.biases.T


class ParentSelection:
    """
    Utility class for selecting parents

    Done in its own class for both readability and separation of concerns
    """

    @staticmethod
    def select_parents(population: List[Cleaner], selection_method="TOURNAMENT") -> Tuple[Cleaner, Cleaner]:
        """
        Selects parents from a population based on a method

        :param population: population of Cleaners to select from
        :param selection_method: string for the method of selection. e.g. "TOURNAMENT" for tournament selection
        :return: Tuple containing two cleaners for the 2 chosen parents
        """
        if selection_method == "TOURNAMENT":
            return ParentSelection.tournament_selection(population)
        elif selection_method == "ROULETTE":
            return ParentSelection.roulette_selection(population)
        else:
            raise Exception(f"Unknown selection_method, value: {selection_method}")

    @staticmethod
    def roulette_selection(population: List[Cleaner]) -> Tuple[Cleaner, Cleaner]:
        """
        Does roulette wheel selection to choose new parents

        :param population: list of Cleaner objects for the population
        :return: a tuple containing the two selected Cleaner parents
        """
        population_fitness = evalFitness(population)

        # Normalise the fitness
        population_fitness = population_fitness / np.max(population_fitness)
        population_fitness_indices_sorted = np.argsort(population_fitness)

        rand_num = np.random.uniform()

        j = 0
        for i in range(len(population_fitness_indices_sorted) - 1):  # Don't go to the end
            j = population_fitness_indices_sorted[i]
            fitness = population_fitness[j]
            if rand_num >= fitness:
                return population[j], population[j + 1]
        return population[j], population[j + 1]

    @staticmethod
    def tournament_selection(population: List[Cleaner], sample_size_factor=0.20) -> Tuple[Cleaner, Cleaner]:
        """
        Does tournament selection to choose new parents

        :param population: population of Cleaner to choose parents from
        :param sample_size_factor: size of the subset to create as a ratio of the total population size
        :return: a tuple containing two Cleaners for the selected parents
        """
        sample_size = int(len(population) * sample_size_factor)
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_fitnesses = evalFitness(sample)

        top_two_parent_indices = np.argpartition(sample_fitnesses, -2)[-2:]

        return sample[top_two_parent_indices]


class CreateOffspring:
    """
    Utility class for creating offspring

    Done in its own class for both readability and separation of concerns
    """

    @staticmethod
    def create_offspring(parent_1: Cleaner, parent_2: Cleaner, weights_method="RUNIF", biases_method="LIN") -> Cleaner:
        """
        Creates the off spring from two parent Cleaners

        Effectively the main method for CreateOffspring

        :param parent_1: the first Cleaner parent
        :param parent_2: the second Cleaner parent
        :param weights_method: method to cross over weights, default is RUNIF for random uniform crossover
        :param:biases_method: method to cross over biases, default is "LIN" for linear crossover, set to None for no crossover TODO how will no crossover this work?
        :return: a new cleaner
        """
        n_actions = parent_1.n_actions
        n_percepts = parent_2.n_percepts

        # Create offspring weights
        if weights_method == "RUNIF":
            new_weights = CreateOffspring.create_runif_weights(parent_1.chromosome.weights, parent_2.chromosome.weights)
        elif weights_method == "K_POINT":
            new_weights = CreateOffspring.create_k_point_weights(parent_1.chromosome.weights,
                                                                 parent_2.chromosome.weights)
        else:
            raise Exception(f"Unknown argument for weights_method, value: {weights_method}")

        # Create offspring biases
        if biases_method == "LIN":
            new_biases = CreateOffspring.create_lin_int_vec(parent_1.chromosome.biases,
                                                            parent_2.chromosome.biases)
        else:
            raise Exception(f"Unknown argument for biases_method, value: {biases_method}")

        new_chromosome = Chromosome(new_weights, new_biases)

        return Cleaner(n_percepts, n_actions, parent_1.grid_size, parent_1.max_turns, new_chromosome)



    @staticmethod
    def create_lin_int_vec(vec_1: np.ndarray, vec_2: np.ndarray, alpha_low=0.45, alpha_high=0.55):
        """
        Performs a random linear interpolation to do cross over of two vectors

        Returns the result of the operation vec1 * alpha + vec * (1 - alpha), where alpha is

        To keep bias values somewhat more stable than weights, this technique can be used

        :param vec_1: first vector
        :param vec_2: second vector
        :param alpha_low: low value for range of values in alpha vector
        :param alpha_high: high value for range of values in alpha vector
        :return: a cross-over of
        """
        num_biases = len(vec_1)
        alpha_vec = np.random.uniform(low=alpha_low, high=alpha_high, size=num_biases)
        ones_vec = np.ones(shape=num_biases)
        return vec_1 * alpha_vec + vec_2 * (ones_vec - alpha_vec)

    @staticmethod
    def create_runif_weights(weights_1: np.ndarray, weights_2: np.ndarray) -> np.ndarray:
        """
        Creates a weights matrix from two parent's weights using random uniform crossover

        :param weights_1: weight matrix belonging to the first parent
        :param weights_2: weight matrix belonging to the second parent
        :return: a new weights matrix
        """
        new_weights = np.zeros_like(weights_1)

        for i in range(len(new_weights)):
            for j in range(len(new_weights[0])):
                rand_num = np.random.rand()
                if rand_num < 0.5:
                    new_weights[i, j] = weights_1[i, j]
                else:
                    new_weights[i, j] = weights_2[i, j]
        return new_weights

    @staticmethod
    def create_k_point_weights(weights_1: np.ndarray, weights_2: np.ndarray, k_points=4) -> np.ndarray:
        """
        Creates a weight offspring from two parents using a k-point crossover

        Does this by crossing over segments of each row, row by row.

        :param parent1: first parent's weight matrix
        :param parent2: second parent's weight matrix
        :return: a new chromosome
        """
        select_parent1 = np.random.choice([True, False])  # Specify whether to start selection from parent1 or parent2
        new_weights = np.zeros_like(weights_1)

        segment_size = len(new_weights[0]) // k_points
        for i in range(len(weights_1)):
            j = 0
            while j < len(weights_1[0]):
                weights_to_select_from = weights_1 if select_parent1 else weights_2

                # Possibly goes beyond end of array, np handles gracefully though
                new_weights[i][j:j + segment_size] = weights_to_select_from[i][j:j + segment_size]

                select_parent1 = not select_parent1
                j += segment_size

        return new_weights

    @staticmethod
    def mutate_weights(weights: np.ndarray, mutation_rate=0.01):
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


def evalFitness(population: List[Cleaner]) -> np.ndarray:
    """
    Evaluates the fitness of a population

    :param population: list of cleaners
    :return: a numpy nd array for the fitness ratings of the population, in the order that they were inputted
    """
    N = len(population)

    # Fitness initialiser for all agents
    fitness = np.zeros(N)

    for n, cleaner in enumerate(population):
        #  cleaner.game_stats['cleaned'] - int, total number of dirt loads picked up
        #  cleaner.game_stats['emptied'] - int, total number of dirt loads emptied at a charge station
        #  cleaner.game_stats['active_turns'] - int, total number of turns the bot was active (non-zero energy)
        #  cleaner.game_stats['successful_actions'] - int, total number of successful actions performed during active
        #                                                  turns
        #  cleaner.game_stats['recharge_count'] - int, number of turns spent at a charging station
        #  cleaner.game_stats['recharge_energy'] - int, total energy gained from the charging station
        #  cleaner.game_stats['visits'] - int, total number of squares visited (visiting the same square twice counts
        #                                      as one visit)

        cleaned, emptied, active_turns, successful_actions, recharge_count, recharge_energy, visits = cleaner.game_stats.values()

        fitness[n] = cleaned * cleaned / visits if visits != 0 else 0

    return fitness


def add_elitism(old_population: List[Cleaner], old_population_fitness: np.ndarray, elitism_proportion=0.2) -> Tuple:
    """
    Adds Elitism to the population

    Extracted into its own method for ease of reading.

    Works by removing Cleaners from the bottom of the population, and replacing them with a portion of the best Cleaners

    :param old_population: list of the old population
    :param old_population_fitness: list containing the fitness ratings of the old population
    :param elitism_proportion: portion of the top cleaners to carry to the next population with elitism.
            E.g. 0.2 is equivalent to the top 20% Cleaners
    :return: a 2-tuple containing, at each index:

        0: A list for the new population to add Cleaners to. This is of the format [None, None, ... , None, ...elite_parents].
            The size is preset, so values can be easily added to it, without expensive array resizing.
            FYI the ... is the spread operator from JS:
            https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Spread_syntax

        1: A list containing the old population of cleaners but with the bottom 20% of the population removed
            according to their fitness. it is from this array that parents can be selected from in the later part of the
            algorithm
    """

    # Convert to a np array for ease of indexing
    old_population_arr = np.array(old_population)

    old_population_size = len(old_population)

    elite_num_parents = int(old_population_size * elitism_proportion)  # Can adjust quotient as need be
    fitness_sorted_indices = numpy.argsort(old_population_fitness)
    elite_parent_indices = fitness_sorted_indices[-elite_num_parents:]

    # print(f"Elite average {np.mean(old_population_fitness[elite_parent_indices])}")

    # Create new population list, add elite parents to start
    new_population = np.concatenate(
        (np.full(old_population_size - elite_num_parents, fill_value=0, dtype=Cleaner),
         old_population_arr[elite_parent_indices]))

    # Effective old population after removing bottom portion to make room for elite parents, want to keep population
    # size consistent
    effective_old_population: np.ndarray = old_population_arr[fitness_sorted_indices[elite_num_parents:]]

    return new_population.tolist(), effective_old_population.tolist()


def newGeneration(old_population: List[Cleaner]) -> Tuple[List[Cleaner], np.ndarray]:
    """
    Creates a new generation of cleaners

    :param old_population: list of Cleaners for the old population of cleaners
    :return: a Tuple containing a list of the new cleaners (same length as old population), and the average fitness of the old population
    """

    old_population_fitness = evalFitness(old_population)

    # Add elitism
    new_population, effective_old_population = add_elitism(old_population, old_population_fitness)

    for i in range(len(effective_old_population)):
        # Select parents
        parent1, parent2 = ParentSelection.select_parents(effective_old_population,
                                                          selection_method="TOURNAMENT")

        # Create offspring from parents
        new_cleaner = CreateOffspring.create_offspring(parent1, parent2, weights_method="RUNIF")

        # Add the new cleaner to the new population
        new_population[i] = new_cleaner

    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(old_population_fitness)

    assert len(old_population) == len(new_population), f"Population sizes are not equal!"  # For my own sanity

    return new_population, avg_fitness
