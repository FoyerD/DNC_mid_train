# DNC
This code implements the paper Deep Neural Crossover: A Multi-Parent Operator That Leverages Gene Correlations: https://dl.acm.org/doi/abs/10.1145/3638529.3654020

As of August 2024, this code only works with the development branch of the Eckity Github repo.
To see a running example of the code, refer to [dnc_runner_eckity.py](dnc_runner_eckity.py).

To use the running example in your own domain the following parts needs to be altered:
```python
    class BinPackingEvaluator(SimpleIndividualEvaluator):

    def __init__(self, n_items, item_weights, bin_capacity, fitness_dict):
```
BinPackingEvaluator defines the fitness objective for the Bin Packing domain as formulated in the original paper. Any other Evaluator the needs to be maximized can be defined.


"DeepNeuralCrossoverConfig" needs to be set the right number of embedding dimensions, which is total number of values a gene can take.

