# Main loop for genetic algorithm processes


import random

from Genetic_algorithm_processes.S1_selection.methods.fitness.fitness_function import FitnessCalculation
from Genetic_algorithm_processes.S1_selection.methods.selection.stochastic_universal_sampling import StochasticUniversalSampling
from Genetic_algorithm_processes.S1_selection.prompt_chain_selection import PromptChainSelection

from Genetic_algorithm_processes.S2_recombination.methods.crossover.N_crossover_multiparent import NCrossoverMultiparent
from Genetic_algorithm_processes.S2_recombination.methods.crossover._model_based_crossover import ModelBasedCrossover
from Genetic_algorithm_processes.S2_recombination.methods.pairing._random_pairing import RandomPairing
from Genetic_algorithm_processes.S2_recombination.prompt_chain_recombination import PromptChainRecombination

from Genetic_algorithm_processes.S3_mutation.methods.synonym_mutation import SynonymMutation
from Genetic_algorithm_processes.S3_mutation.methods.shuffle_mutation import ShuffleMutation
from Genetic_algorithm_processes.S3_mutation.methods.delete_mutation import DeleteMutation
from Genetic_algorithm_processes.S3_mutation.prompt_chain_mutation import PromptChainMutation

from Genetic_algorithm_processes.S4_migration.methods.topology.model_based_topology import ModelBasedTopology
from Genetic_algorithm_processes.S4_migration.prompt_chain_migration import PromptChainMigration

from Genetic_algorithm_processes.S5_simulated_annealing.simulated_annealing import SimulatedAnnealing

from Genetic_algorithm_processes.S6_replacement.methods.replacement.offspring_fitness_replacement import OffspringFitnessReplacement

from Genetic_algorithm_processes.ollama_run import PromptChainRunner

prompt_chain_population = [
    [("gpt-3.5-turbo", "This is an ", "input prompt"), 
        ("gpt-4", "This ", "another ", "input ", "prompt")],
    [("gpt-4", "Different input prompt here"), 
        ("gpt-3.5-turbo", "", "Yet another prompt input")]
]
population_cap = 100
initial_input = "What is the capital of France?"
solution_output = "Paris"

runner = PromptChainRunner()
runner.update_model_registry()

def run_prompt_chain(prompt_chain, initial_input=initial_input):
    return runner.run_prompt_chain(prompt_chain, initial_input, verbose=False)


# Instances
fitness_algorithm = FitnessCalculation(accuracy_weight=1.0, speed_weight=1.0, token_limit_weight=1.0, accuracy_scaling_factor=0.6, accuracy_testing_model=None, speed_limit=10.0, token_hard_limit_threshold=1, token_soft_limit_threshold=0.7)
selection_algorithm = StochasticUniversalSampling(selection_ratio=0.5)
replacement_algorithm = OffspringFitnessReplacement(fitness_function=fitness_algorithm)

prompt_chain_pairing_instance = RandomPairing(size_of_pairs=2, number_of_pairs=2)
prompt_chain_crossover_instance = NCrossoverMultiparent(crossover_num=2, distribution=lambda length: random.randint(1, max(1, length - 1)), number_offsprings=2),
prompt_crossover_instance = ModelBasedCrossover(crossover_method=prompt_chain_crossover_instance)

synonym_mutation = SynonymMutation(mutation_rate=0.1, pos_tags_to_mutate=['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'])
shuffle_mutation = ShuffleMutation(N_segment_cut=None, N_distribution=lambda length: random.randint(1, max(1, length - 1)), shuffling_constant=1.0)
delete_mutation = DeleteMutation(min_segment_fraction=0.1, max_segment_fraction=0.3)

model_based_migration = ModelBasedTopology()


# 1. Selection
prompt_chain_selection = PromptChainSelection(selection_algorithm=selection_algorithm, replacement_algorithm=replacement_algorithm, fitness_algorithm=fitness_algorithm, population_cap=population_cap)
selected, population_fitness = prompt_chain_selection.evauluate_select_prompt_chains(prompt_chain_population, run_prompt_chain, solution_output)
print(f"Selected Prompt Chains: {selected}")

# 2. Recombination
prompt_chain_recombination = PromptChainRecombination(run_prompt_chain)
prompt_chain_offsprings = prompt_chain_recombination.recombine_prompt_chains(selected)
print(f"Prompt Chain Offsprings: {prompt_chain_offsprings}")

# 3. Mutation
mutation_instance = PromptChainMutation(mutation_chance=0.1, mutation_methods=[synonym_mutation.mutate, shuffle_mutation.mutate, delete_mutation.mutate])
prompt_chain_offsprings = mutation_instance.mutate_population(prompt_chain_offsprings)
print(f"Mutated Offsprings population: {prompt_chain_offsprings}")

# 4. Migration
migration_instance = PromptChainMigration(migration_chance=0.01)
prompt_chain_offsprings = migration_instance.migrate_population(prompt_chain_offsprings)
print(f"Migrated Population: {prompt_chain_offsprings}")

# Next iteration - reselect new population from offsprings
selected, population_fitness = prompt_chain_selection.evauluate_select_prompt_chains(prompt_chain_offsprings, run_prompt_chain, solution_output)
prompt_chain_offsprings = prompt_chain_recombination.recombine_prompt_chains(selected)
prompt_chain_offsprings = mutation_instance.mutate_population(prompt_chain_offsprings)
prompt_chain_offsprings = migration_instance.migrate_population(prompt_chain_offsprings)
print(f"Next generation population: {prompt_chain_offsprings}")


# 5. Simulated Annealing
print(f"\n\n========== STARTING SIMULATED ANNEALING OPTIMIZATION ==========")
simulated_annealing_instance = SimulatedAnnealing(
    run_prompt_chain_func=run_prompt_chain,
    steps=10,
    initial_temp=100.0,
    elite_selection_ratio=0.1,
    cooling_schedule='exponential',
    cooling_rate=0.95
)
optimized_population = simulated_annealing_instance.simulate_annealing(prompt_chain_population)
print(f"Optimized Population after Simulated Annealing: {optimized_population}")
print(f"\nOptimized population size: {len(optimized_population)}")
print(f"Best fitness found: {simulated_annealing_instance.best_fitness:.4f}")
print(f"\nBest solution:")
for i, step in enumerate(simulated_annealing_instance.best_prompt_chain):
    print(f"  Step {i+1}: Model={step[0]}, Segments={step[1:]}")

print(f"\nFitness history (last 10): {simulated_annealing_instance.fitness_history[-10:]}")