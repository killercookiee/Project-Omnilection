"""
Simulated Annealing for Prompt Chain Optimization

This module implements a simulated annealing algorithm to optimize prompt chains
by exploring the prompt chain population through temperature-based acceptance criteria.
"""

import random
import math
import copy
from typing import List, Tuple

from Genetic_algorithm_processes.S1_selection.methods.fitness.fitness_function import FitnessCalculation
from Genetic_algorithm_processes.S3_mutation.prompt_chain_mutation import PromptChainMutation



class SimulatedAnnealing:
    """
    Simulated Annealing optimizer for prompt chain populations.
    
    The algorithm:
    1. Starts with a population of prompt chains
    2. Selects elite individuals to keep
    3. For remaining slots, uses simulated annealing to generate new candidates:
       - Mutates current prompt chains
       - Accepts improvements always
       - Accepts worse prompt chains with probability based on temperature
    4. Temperature decreases over time (cooling schedule)
    
    Parameters:
    -----------
    run_prompt_chain_func : Callable
        Function that executes a prompt chain and returns outputs with timing
        Signature: prompt_chain -> [(output, time_taken), ...]
    
    steps : int
        Number of simulated annealing iterations per slot
    
    initial_temp : float
        Starting temperature for the annealing process
        Higher values allow more exploration initially
    
    elite_selection_ratio : float
        Ratio of top performers to keep unchanged (0.0 to 1.0)
        e.g., 0.2 keeps top 20% as elite
    
    fitness_function = FitnessCalculation()
        Custom fitness function for scoring prompt chains
    
    mutation_function = PromptChainMutation(mutation_chance=1.0)
        Custom mutation function for generating neighbors
    
    cooling_schedule : str, optional
        Cooling schedule type: 'exponential', 'linear', or 'logarithmic'
        Default: 'exponential'
    
    cooling_rate : float, optional
        Rate of temperature decrease (for exponential/linear schedules)
        Default: 0.95
    """
    
    def __init__(self,
        run_prompt_chain_func: callable,
        steps: int = 10,
        initial_temp: float = 100.0,
        elite_selection_ratio: float = 0.2,
        fitness_function = FitnessCalculation(),
        mutation_function = PromptChainMutation(mutation_chance=1.0),
        cooling_schedule: str = 'exponential',
        cooling_rate: float = 0.95
    ):
        self.run_prompt_chain_func = run_prompt_chain_func
        self.steps = steps
        self.initial_temp = initial_temp
        self.elite_selection_ratio = elite_selection_ratio
        self.cooling_schedule = cooling_schedule
        self.cooling_rate = cooling_rate
        
        self.fitness_function = fitness_function
        self.mutation_function = mutation_function
        
        # Track best prompt chain found
        self.best_prompt_chain = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def calculate_fitness(self, prompt_chain: List[Tuple], eval_context: dict) -> float:
        """
        Calculate fitness score for a prompt chain.
        
        Parameters:
        -----------
        prompt_chain : List[Tuple]
            Prompt chain to evaluate
        
        Returns:
        --------
        float : Fitness score
        """
        if self.fitness_function is None:
            # Simple fallback: negative total time
            prompt_output_chain = self.run_prompt_chain_func(prompt_chain)
            total_time = sum([time_taken for _, time_taken in prompt_output_chain])
            return -total_time
        
        # Use provided fitness function
        prompt_output_chain = self.run_prompt_chain_func(prompt_chain)
        
        return self.fitness_function.evaluate_prompt_chain(
            prompt_chain,
            prompt_output_chain,
            eval_context 
        )
    
    def get_temperature(self, current_step: int) -> float:
        """
        Calculate temperature based on cooling schedule.
        
        Parameters:
        -----------
        current_step : int
            Current iteration step
        
        Returns:
        --------
        float : Current temperature
        """
        if self.cooling_schedule == 'exponential':
            return self.initial_temp * (self.cooling_rate ** current_step)
        
        elif self.cooling_schedule == 'linear':
            return self.initial_temp - (self.cooling_rate * current_step)
        
        elif self.cooling_schedule == 'logarithmic':
            return self.initial_temp / (1 + self.cooling_rate * math.log(1 + current_step))
        
        else:
            # Default to exponential
            return self.initial_temp * (self.cooling_rate ** current_step)
    
    def acceptance_probability(self, current_fitness: float, new_fitness: float, temperature: float) -> float:
        """
        Calculate probability of accepting a worse prompt_chain.
        
        Uses the Metropolis criterion: exp(ΔE / T)
        
        Parameters:
        -----------
        current_fitness : float
            Fitness of current prompt_chain
        new_fitness : float
            Fitness of new candidate prompt_chain
        temperature : float
            Current temperature
        
        Returns:
        --------
        float : Acceptance probability (0.0 to 1.0)
        """
        if new_fitness > current_fitness:
            return 1.0  # Always accept improvements
        
        if temperature <= 0:
            return 0.0  # Never accept worse prompt chains at zero temperature
        
        delta = new_fitness - current_fitness
        return math.exp(delta / temperature)
    
    def mutate_prompt_chain(self, prompt_chain: List[Tuple]) -> List[Tuple]:
        """
        Generate a neighboring prompt chains through mutation.
        
        Parameters:
        -----------
        prompt_chain : List[Tuple]
            Current prompt chain
        
        Returns:
        --------
        List[Tuple] : Mutated prompt chain
        """
        if self.mutation_function is None:
            # Simple fallback: return a copy
            return copy.deepcopy(prompt_chain)
        
        # Use provided mutation function
        mutated_population = self.mutation_function.mutate_prompt_chain(prompt_chain)
        return mutated_population
    
    def optimize_single_prompt_chain(self, initial_prompt_chain: List[Tuple], eval_context: dict) -> Tuple[List[Tuple], float]:
        """
        Run simulated annealing on a single prompt chain.
        
        Parameters:
        -----------
        initial_prompt_chain : List[Tuple]
            Starting prompt chain
        
        Returns:
        --------
        Tuple[List[Tuple], float] : Best prompt chain found and its fitness
        """
        current_prompt_chain = copy.deepcopy(initial_prompt_chain)
        current_fitness = self.calculate_fitness(current_prompt_chain, eval_context)
        
        best_prompt_chain = copy.deepcopy(current_prompt_chain)
        best_fitness = current_fitness
        for step in range(self.steps):
            # Generate neighbor
            new_prompt_chain = self.mutate_prompt_chain(current_prompt_chain)
            new_fitness = self.calculate_fitness(new_prompt_chain, eval_context)
            
            # Calculate acceptance probability
            temperature = self.get_temperature(step)
            accept_prob = self.acceptance_probability(current_fitness, new_fitness, temperature)
            
            # Accept or reject
            if random.random() < accept_prob:
                current_prompt_chain = new_prompt_chain
                current_fitness = new_fitness
                
                # Update best if improved
                if current_fitness > best_fitness:
                    best_prompt_chain = copy.deepcopy(current_prompt_chain)
                    best_fitness = current_fitness
        
        return best_prompt_chain, best_fitness
    
    def optimize_single_record(self, initial_record: dict, eval_context: dict) -> Tuple[dict, float]:
            """
            Runs SA on a genetic envelope (OffspringRecord). 
            """
            current_record = copy.deepcopy(initial_record)
            current_fitness = self.calculate_fitness(current_record["chain"], eval_context)
            
            best_record = copy.deepcopy(current_record)
            best_fitness = current_fitness
            
            for step in range(self.steps):
                # Mutate purely the chain, package back into the envelope
                new_chain = self.mutate_prompt_chain(current_record["chain"])
                new_fitness = self.calculate_fitness(new_chain, eval_context)
                
                temperature = self.get_temperature(step)
                accept_prob = self.acceptance_probability(current_fitness, new_fitness, temperature)
                
                if random.random() < accept_prob:
                    current_record["chain"] = new_chain
                    current_record["metadata"]["sa_optimized"] = True
                    current_fitness = new_fitness
                    
                    if current_fitness > best_fitness:
                        best_record = copy.deepcopy(current_record)
                        best_fitness = current_fitness
            
            return best_record, best_fitness

    def simulate_annealing(self, offspring_records: List[dict], eval_context: dict) -> List[dict]:
        """
        Applies SA to an entire population of OffspringRecords.
        """
        if not offspring_records:
            return []
        
        population_size = len(offspring_records)
        elite_size = max(1, int(population_size * self.elite_selection_ratio))
        
        pop_with_fitness = []
        for record in offspring_records:
            fitness = self.calculate_fitness(record["chain"], eval_context)
            pop_with_fitness.append((record, fitness))
            
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        elite_records = [rec for rec, _ in pop_with_fitness[:elite_size]]
        non_elite_records = [rec for rec, _ in pop_with_fitness[elite_size:]]

        optimized_elite = []
        for elite_rec in elite_records:
            opt_rec, opt_fitness = self.optimize_single_record(elite_rec, eval_context)
            optimized_elite.append(opt_rec)
            
            self.fitness_history.append(opt_fitness)
            if opt_fitness > self.best_fitness:
                self.best_prompt_chain = copy.deepcopy(opt_rec["chain"])
                self.best_fitness = opt_fitness
                
        return optimized_elite + non_elite_records
    
    def get_best_prompt_chain(self) -> Tuple[List[Tuple], float]:
        """
        Get the best prompt chain found during optimization.
        
        Returns:
        --------
        Tuple[List[Tuple], float] : Best prompt chain and its fitness
        """
        return self.best_prompt_chain, self.best_fitness
    
    def get_fitness_history(self) -> List[float]:
        """
        Get history of fitness scores from optimization.
        
        Returns:
        --------
        List[float] : Fitness scores over time
        """
        return self.fitness_history


if __name__ == "__main__":
    # Example usage
    print("Simulated Annealing for Prompt Chain Optimization")
    print("=" * 60)
    
    # Mock run_prompt_chain function
    def run_prompt_chain(prompt_chain, initial_input):
        # Output: [(prompt_output, model_run_info), ...]
        return [("The capital of France is Paris.", {"total_duration": 2.5, "load_duration": 0.5, "prompt_eval_count": 1, "prompt_eval_duration": 0.5, "prompt_eval_rate": 2.0, "eval_count": 1, "eval_duration": 2.0, "eval_rate": 0.5}),
                ("The capital of Germany is Berlin.", {"total_duration": 2.067, "load_duration": 0.859, "prompt_eval_count": 14, "prompt_eval_duration": 0.208, "prompt_eval_rate": 50.30, "eval_count": 14, "eval_duration": 2.92, "eval_rate": 4.79})]
    
    # Create initial population
    initial_population = [
        [("gpt-3.5-turbo", "This is an ", "input prompt"), 
         ("gpt-4", "This ", "another ", "input ", "prompt")],
        [("gpt-4", "Different input prompt here"), 
         ("gpt-3.5-turbo", "", "Yet another prompt input")],
        [("gpt-3.5-turbo", "Sample prompt one"), 
         ("gpt-4", "Sample prompt two")],
        [("gpt-4", "First prompt segment"), 
         ("gpt-4", "Second prompt segment")],
        [("gpt-3.5-turbo", "Only one prompt in this chain")]
    ]
    initial_input = "What is the capital of France?"
    solution_output = "Paris"
    
    # Create simulated annealing instance
    sa = SimulatedAnnealing(
        run_prompt_chain_func=run_prompt_chain,
        steps=10,
        initial_temp=100.0,
        elite_selection_ratio=0.2,
        cooling_schedule='exponential',
        cooling_rate=0.95
    )
    
    print(f"Initial population size: {len(initial_population)}")
    print(f"Elite selection ratio: {sa.elite_selection_ratio}")
    print(f"Simulated annealing steps per prompt chain: {sa.steps}")
    print(f"Initial temperature: {sa.initial_temp}")
    print(f"Cooling schedule: {sa.cooling_schedule}")
    print()
    
    # Run optimization
    print("Running simulated annealing optimization...")
    optimized_population = sa.simulate_annealing(initial_population, eval_context={"initial_input": initial_input, "solution_output": solution_output})
    
    print(f"\nOptimized population size: {len(optimized_population)}")
    print(f"Best fitness found: {sa.best_fitness:.4f}")
    print(f"\nBest prompt chain:")
    for i, step in enumerate(sa.best_prompt_chain):
        print(f"  Step {i+1}: Model={step[0]}, Segments={step[1:]}")
    
    print(f"\nFitness history (last 10): {sa.fitness_history[-10:]}")

    print("\n\nFinal optimized population: ", optimized_population)