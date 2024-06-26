from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import IntVectorNPointMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.genetic_encodings.ga.int_vector import IntVector
import json
import numpy as np


class GAIntegerStringVectorCreator(GAVectorCreator):
    def __init__(self,
                 length=1,
                 bounds=(0, 1),
                 gene_creator=None,
                 events=None):
        super().__init__(length=length, bounds=bounds, gene_creator=gene_creator, vector_type=IntVector,
                         events=events)


class BinPackingEvaluator(SimpleIndividualEvaluator):

    def __init__(self, n_items, item_weights, bin_capacity, fitness_dict):
        super().__init__()
        self.n_items = n_items
        self.item_weights = item_weights
        self.bin_capacity = bin_capacity
        self.fitness_dict = fitness_dict

    def evaluate_individual(self, individual):
        """
            Compute the fitness value of a given individual.

            Parameters
            ----------
            individual: Vector
                The individual to compute the fitness value for.

            Returns
            -------
            float
                The evaluated fitness value of the given individual.
        """
        return self.get_bin_packing_fitness(np.array(individual.vector))

    def get_bin_packing_fitness(self, individual, penalty=100):
        fitness_dict = self.fitness_dict

        if tuple(individual) in fitness_dict:
            return fitness_dict[tuple(individual)]

        fitness = 0
        bin_capacities = np.zeros(self.n_items)
        legal_solution = True

        for item_index, bin_index in enumerate(individual):
            bin_capacities[bin_index] += self.item_weights[item_index]

            if bin_capacities[bin_index] > self.bin_capacity:
                legal_solution = False
                fitness -= penalty

        if legal_solution:
            utilized_bins = bin_capacities[bin_capacities > 0]
            fitness = ((bin_capacities / self.bin_capacity) ** 2).sum() / len(utilized_bins)

        fitness_dict[tuple(individual)] = fitness
        return fitness


def main():
    fitness_dict = {}
    datasets_json = json.load(open('./datasets/hard_parsed.json', 'r'))
    dataset_name = 'BPP_14'
    dataset_item_weights = np.array(datasets_json[dataset_name]['items'])
    dataset_bin_capacity = datasets_json[dataset_name]['max_bin_weight']
    dataset_n_items = len(dataset_item_weights)

    ind_length = dataset_n_items
    min_bound, max_bound = 0, dataset_n_items - 1

    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=GAIntegerStringVectorCreator(length=ind_length, bounds=(min_bound, max_bound)),
                      population_size=100,
                      # user-defined fitness evaluation method
                      evaluator=BinPackingEvaluator(n_items=dataset_n_items, item_weights=dataset_item_weights,
                                                    bin_capacity=dataset_bin_capacity, fitness_dict=fitness_dict),
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
                      # elitism_rate=1 / 300,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          VectorKPointsCrossover(probability=0.5, k=1),
                          IntVectorNPointMutation(probability=0.1, n=ind_length)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=5, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=6000,
        # termination_checker=ThresholdFromTargetTerminationChecker(optimal=100, threshold=0.0),
        statistics=BestAverageWorstStatistics()
    )

    # evolve the generated initial population
    algo.evolve()

    # Execute (show) the best solution
    print(algo.execute())


if __name__ == '__main__':
    main()
