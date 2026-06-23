"""
Genetic_algorithm_processes/S5_simulated_annealing/simulated_annealing.py
"""

import random
import math
import copy
import hashlib
from typing import List

from Genetic_algorithm_processes.Data.general_datamanager import GeneralDataManager
from Genetic_algorithm_processes.S5_simulated_annealing.methods.micro_mutation import MicroMutation

def get_chain_id(chain: list) -> str:
    return f"chain_{hashlib.md5(str(chain).encode('utf-8')).hexdigest()[:12]}"

class SimulatedAnnealing:
    def __init__(self,
        evaluator_func: callable,
        micro_mutator: MicroMutation,
        gdm: GeneralDataManager,
        steps: int = 3,           # Keep this low (3-5) because it runs sequentially!
        initial_temp: float = 10.0,
        elite_selection_ratio: float = 0.2, # Optimize the top 20% of the population
        cooling_rate: float = 0.80,
        verbose: bool = False
    ):
        self.evaluator_func = evaluator_func
        self.micro_mutator = micro_mutator
        self.gdm = gdm
        self.steps = steps
        self.initial_temp = initial_temp
        self.elite_selection_ratio = elite_selection_ratio
        self.cooling_rate = cooling_rate
        self.verbose = verbose

    def acceptance_probability(self, current_fitness: float, new_fitness: float, temperature: float) -> float:
        if new_fitness > current_fitness:
            return 1.0 
        if temperature <= 0.01:
            return 0.0 
        delta = new_fitness - current_fitness
        # Guard against math overflow
        try:
            return math.exp(delta / temperature)
        except OverflowError:
            return 0.0

    def optimize_single_record(self, record: tuple, generation_num: int, task: dict) -> tuple:
        """
        Takes an evaluated record format: (chain_id, chain, fitness, metadata)
        Returns an optimized evaluated record.
        """
        current_id, current_chain, current_fitness, current_metadata = record
        
        best_record = record
        best_fitness = current_fitness
        
        temperature = self.initial_temp

        if self.verbose:
            print(f"\n  [SA] Optimizing Elite: {current_id} (Starting Fit: {current_fitness:.4f})")

        for step in range(self.steps):
            # 1. Micro-Mutate based on current temperature
            new_chain = self.micro_mutator.mutate(current_chain, temperature, self.initial_temp)
            
            # If the mutator failed to make a change, skip evaluation
            if new_chain == current_chain:
                temperature *= self.cooling_rate
                continue
                
            new_id = get_chain_id(new_chain)
            
            # 2. Register intermediary so the database tracks the tweak
            new_meta = current_metadata.copy()
            new_meta["sa_optimized"] = True
            new_meta["sa_temperature"] = temperature
            self.gdm.register_intermediary_chain(new_id, new_chain, parents=[current_id], metadata=new_meta)
            
            # 3. Evaluate the new chain using the global cache (NOW PASSING TASK)
            eval_packet = [{"chain_id": new_id, "chain": new_chain, "parents": [current_id], "metadata": new_meta}]
            evaluated_result = self.evaluator_func(eval_packet, generation_num, task)
            
            # Extract the new fitness
            new_record = evaluated_result[0]
            new_fitness = new_record[2]
            
            # 4. Acceptance Check
            accept_prob = self.acceptance_probability(current_fitness, new_fitness, temperature)
            
            if self.verbose:
                status = "✅ ACCEPTED (Improved)" if new_fitness > current_fitness else ("⚠️ ACCEPTED (Exploration)" if random.random() < accept_prob else "❌ REJECTED")
                print(f"       Step {step+1} | Temp {temperature:.1f} | Fit: {new_fitness:.4f} | {status}")

            if new_fitness > current_fitness or random.random() < accept_prob:
                current_chain = new_chain
                current_id = new_id
                current_fitness = new_fitness
                current_metadata = new_record[3]
                
                if current_fitness > best_fitness:
                    best_record = new_record
                    best_fitness = current_fitness
            
            # 5. Cool down
            temperature *= self.cooling_rate

        return best_record

    def process_population(self, evaluated_population: list[tuple], generation_num: int, task: dict) -> list[tuple]:
        """
        Takes the full evaluated population, picks the elites, runs SA on them, 
        and returns the combined new population.
        """
        if not evaluated_population:
            return []
            
        # Sort by fitness
        sorted_pop = sorted(evaluated_population, key=lambda x: x[2], reverse=True)
        
        elite_count = max(1, int(len(sorted_pop) * self.elite_selection_ratio))
        elites = sorted_pop[:elite_count]
        the_rest = sorted_pop[elite_count:]
        
        optimized_elites = []
        for elite in elites:
            # NOW PASSING TASK TO OPTIMIZER
            optimized_record = self.optimize_single_record(elite, generation_num, task)
            optimized_elites.append(optimized_record)
            
        return optimized_elites + the_rest