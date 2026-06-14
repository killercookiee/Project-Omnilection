# Main loop for genetic algorithm processes

import random
import uuid
import hashlib
from datetime import datetime, timezone

# ── Data management ────────────────────────────────────────────────────────────
from Data.general_datamanager import GeneralDataManager

# ── Dataset / population seeding ───────────────────────────────────────────────
from Dataset_Prompts.gene_pool_manager import GenePoolManager
from Dataset_Prompts.initial_population_generator import InitialPopulationGenerator

# ── GA pipeline ────────────────────────────────────────────────────────────────
from Genetic_algorithm_processes.S1_selection.methods.fitness.fitness_function import FitnessCalculation
from Genetic_algorithm_processes.S1_selection.methods.selection.stochastic_universal_sampling import StochasticUniversalSampling
from Genetic_algorithm_processes.S1_selection.prompt_chain_selection import PromptChainSelection

from Genetic_algorithm_processes.S2_recombination.methods.pairing.lineage_pairing import LineagePairing
from Genetic_algorithm_processes.S2_recombination.methods.crossover.lineage_crossover import LineageCrossover
from Genetic_algorithm_processes.S2_recombination.methods.crossover.model_based_crossover import ModelBasedCrossover
from Genetic_algorithm_processes.S2_recombination.prompt_chain_recombination import PromptChainRecombination

from Genetic_algorithm_processes.S3_mutation.methods.synonym_mutation import SynonymMutation
from Genetic_algorithm_processes.S3_mutation.methods.shuffle_mutation import ShuffleMutation
from Genetic_algorithm_processes.S3_mutation.methods.delete_mutation import DeleteMutation
from Genetic_algorithm_processes.S3_mutation.methods.semantic_llm_mutation import SemanticLLMMutation
from Genetic_algorithm_processes.S3_mutation.prompt_chain_mutation import PromptChainMutation

from Genetic_algorithm_processes.S4_migration.methods.topology.model_based_topology import ModelBasedTopology
from Genetic_algorithm_processes.S4_migration.prompt_chain_migration import PromptChainMigration

from Genetic_algorithm_processes.S6_replacement.methods.replacement.lineage_replacement import LineageReplacement
from Genetic_algorithm_processes.S6_replacement.prompt_chain_replacement import PromptChainReplacement

from Genetic_algorithm_processes.ollama_run import PromptChainRunner


# ── Configuration & Data Management ────────────────────────────────────────────

RESUME_FROM_SAVE = False  
TARGET_SAVE_DIR  = None   

GENERATIONS = 10
population_cap  = 100
initial_input   = "What is the capital of France?"
solution_output = "Paris"

gdm = GeneralDataManager(run_dir=TARGET_SAVE_DIR, resume_from_save=RESUME_FROM_SAVE)

runner = PromptChainRunner(verbose=False)
runner.update_model_registry(benchmark=False)

def get_chain_id(chain: list) -> str:
    return f"chain_{hashlib.md5(str(chain).encode('utf-8')).hexdigest()[:12]}"


# ── Core Execution & Evaluation Logic ──────────────────────────────────────────

def evaluate_population_with_cache(population_records: list[dict], generation_num: int) -> list[tuple]:
    evaluated_data = []
    final_records = []
    
    heritage_db = gdm.heritage_data_manager.local_heritage_database.get("prompt_chains", {})
    
    for record in population_records:
        chain_id = record["chain_id"]
        chain = record["chain"]
        parents = record.get("parents", [])
        metadata = record.get("metadata", {})
        metadata["generation"] = generation_num

        heritage_record = heritage_db.get(chain_id, {})
        cached_fitness = heritage_record.get("fitness")
        has_been_evaluated = cached_fitness is not None

        if has_been_evaluated:
            existing_meta = heritage_record.get("metadata", {})
            existing_meta["generation"] = generation_num
            final_records.append((chain_id, chain, cached_fitness, existing_meta))
            print(f"  [Cache Hit] Skipped execution for: {chain_id}")
            gdm.sync_population([(chain_id, chain, cached_fitness, existing_meta)])
        else:
            prompt_output_chain = runner.run_prompt_chain(chain, initial_input)
            evaluated_data.append((chain_id, chain, prompt_output_chain, parents, metadata))

    if evaluated_data:
        print(f"  [Execution] Evaluating fitness for {len(evaluated_data)} new chains...")
        chains_and_outputs = [(item[1], item[2]) for item in evaluated_data]
        
        # Unpack the new 3-part return from evaluate_population
        new_evaluations = fitness_algorithm.evaluate_population(chains_and_outputs, initial_input, solution_output)
        
        for i, (chain, fitness, telemetry) in enumerate(new_evaluations):
            chain_id = evaluated_data[i][0]
            parents = evaluated_data[i][3]
            metadata = evaluated_data[i][4]
            
            # ── Inject Telemetry into Metadata ──
            metadata.update(telemetry)
            metadata["creation_time"] = datetime.now(timezone.utc).isoformat()
            
            final_records.append((chain_id, chain, fitness, metadata))
            
    gdm.sync_population(final_records)
    return final_records


# ── GA operator instances ──────────────────────────────────────────────────────

fitness_algorithm = FitnessCalculation()
selection_algorithm = PromptChainSelection(verbose=True)

recombination_algorithm = PromptChainRecombination(
    prompt_chain_pairing_instance=LineagePairing(verbose=True),
    prompt_chain_crossover_instance=LineageCrossover(verbose=True),
    prompt_crossover_instance=ModelBasedCrossover(verbose=False),
    gdm=gdm
)

mutation_algorithm = PromptChainMutation(
    base_mutation_chance=0.10,
    gdm=gdm,
    verbose=True,
    mutation_methods=[
        # Use your fast model for mutations. Make sure this model is pulled in Ollama!
        SemanticLLMMutation(runner=runner, mutator_model="qwen2.5-coder:0.5b", verbose=False).mutate,
        
        # You can still include one fast structural mutation if you want, 
        # but weighting Semantic higher gives you those big genetic leaps.
        DeleteMutation(min_segment_fraction=0.1, max_segment_fraction=0.3).mutate,
    ]
)

migration_algorithm = PromptChainMigration(migration_chance=0.01)

replacement_algorithm = PromptChainReplacement(
    replacement_algorithm=LineageReplacement(population_cap=population_cap, verbose=True),
    gdm=gdm 
)


# ── Population Initialization / Resumption ─────────────────────────────────────

current_population_records = []
start_generation = 0
saved_pop_data = gdm.population_data_manager.local_population_data

if RESUME_FROM_SAVE and saved_pop_data and saved_pop_data.get("population"):
    print(f"\n[main] 💾 Found saved population data in {gdm.run_dir}. Resuming...")
    saved_pop = saved_pop_data["population"]
    saved_meta = saved_pop_data.get("metadata", {})
    
    for entry in saved_pop:
        chain_id = entry[0]
        chain_tuples = [tuple(step) for step in entry[1]]
        fitness = entry[2]
        metadata = entry[3]
        current_population_records.append((chain_id, chain_tuples, fitness, metadata))
    
    start_generation = saved_meta.get("generation", 0) + 1
    print(f"[main] 🚀 Resuming execution at Generation {start_generation}.")
else:
    print("\n[main] 🌱 Generating initial population...")
    gene_pool_manager = GenePoolManager()
    segments = gene_pool_manager.load_prompt_segments()
    if not segments:
        gene_pool_manager.run_pipeline()
        segments = gene_pool_manager.load_prompt_segments()

    available_models: list[str] = list(runner.model_registry.keys())
    initial_pop_generator = InitialPopulationGenerator(
        segments=segments, available_models=available_models, population_cap=population_cap
    )
    
    raw_chains = initial_pop_generator.generate()
    initial_dicts = []
    for chain in raw_chains:
        chain_id = get_chain_id(chain)
        metadata = {"prefix_len": 1}
        # Register as intermediary first
        gdm.register_intermediary_chain(chain_id, chain, [], metadata)
        initial_dicts.append({
            "chain_id": chain_id,
            "chain": chain,
            "parents": [],  
            "metadata": metadata
        })
        
    print("\n[main] 🧬 Evaluating Initial Population (Generation 0)...")
    current_population_records = evaluate_population_with_cache(initial_dicts, generation_num=0)
    start_generation = 1


# ── GA main loop ───────────────────────────────────────────────────────────────

for gen in range(start_generation, GENERATIONS):
    print(f"\n\n{'='*60}")
    print(f"=== GENERATION {gen} ===")
    print(f"{'='*60}")

    mutation_algorithm.adjust_mutation_rate(current_population_records)

    selected_records = selection_algorithm.select_prompt_chains(current_population_records)
    offspring_dicts = recombination_algorithm.recombine_prompt_chains(selected_records)
    offspring_dicts = mutation_algorithm.mutate_population(offspring_dicts)
    offspring_dicts = migration_algorithm.migrate_population(offspring_dicts)

    print(f"\n--- Evaluating Generation {gen} Offspring ---")
    evaluated_offspring_records = evaluate_population_with_cache(offspring_dicts, generation_num=gen)

    current_population_records = replacement_algorithm.replace_population(
        current_population_records, evaluated_offspring_records
    )

print("\n\n========== EVOLUTION COMPLETE ==========")