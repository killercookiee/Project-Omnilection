# Main loop for genetic algorithm processes

import os
import sys
import time
import hashlib
import random
import logging
import signal
from datetime import datetime, timezone

# ── Force Hardware Constraints BEFORE loading Ollama ───────────────────────────
def _find_local_folder(folder_name: str = "Project-Omnilection") -> str:
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir and current_dir != '/':
        if os.path.basename(current_dir) == folder_name:
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Could not find the project root.")

def _set_directory():
    target_folder = _find_local_folder()
    if os.path.realpath(os.getcwd()) != os.path.realpath(target_folder):
        os.chdir(target_folder)
    if target_folder not in sys.path:
        sys.path.insert(0, target_folder)

_set_directory()

# ── Data management & Pipeline Imports ─────────────────────────────────────────
from Data.general_datamanager import GeneralDataManager
from Dataset_Prompts.gene_pool_manager import GenePoolManager
from Dataset_Prompts.initial_population_generator import InitialPopulationGenerator

from Genetic_algorithm_processes.S1_selection.methods.fitness.fitness_function import FitnessCalculation
from Genetic_algorithm_processes.S1_selection.methods.selection.stochastic_universal_sampling import StochasticUniversalSampling
from Genetic_algorithm_processes.S1_selection.prompt_chain_selection import PromptChainSelection

from Genetic_algorithm_processes.S2_recombination.methods.pairing.lineage_pairing import LineagePairing
from Genetic_algorithm_processes.S2_recombination.methods.crossover.lineage_crossover import LineageCrossover
from Genetic_algorithm_processes.S2_recombination.methods.crossover.model_based_crossover import ModelBasedCrossover
from Genetic_algorithm_processes.S2_recombination.prompt_chain_recombination import PromptChainRecombination

from Genetic_algorithm_processes.S3_mutation.methods.delete_mutation import DeleteMutation
from Genetic_algorithm_processes.S3_mutation.methods.semantic_llm_mutation import SemanticLLMMutation
from Genetic_algorithm_processes.S3_mutation.methods.gene_pool_mutation import GenePoolSearchMutation
from Genetic_algorithm_processes.S3_mutation.prompt_chain_mutation import PromptChainMutation

from Genetic_algorithm_processes.S4_migration.prompt_chain_migration import PromptChainMigration

from Genetic_algorithm_processes.S5_simulated_annealing.simulated_annealing import SimulatedAnnealing
from Genetic_algorithm_processes.S5_simulated_annealing.methods.micro_mutation import MicroMutation

from Genetic_algorithm_processes.S6_replacement.methods.replacement.lineage_replacement import LineageReplacement
from Genetic_algorithm_processes.S6_replacement.prompt_chain_replacement import PromptChainReplacement

from Genetic_algorithm_processes.ollama_run import PromptChainRunner

# ── Force Hardware Constraints & Suppress Mac Warnings ─────────────────────────
os.environ["OLLAMA_NUM_THREADS"] = "6"
os.environ["MallocStackLogging"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("MallocStackLogging", None)

# ── Global Interrupt Manager ───────────────────────────────────────────────────
INTERRUPT_STATE = {
    "last_press_time": 0.0,
    "skip_cooldown": False
}

def handle_sigint(signum, frame):
    current_time = time.time()
    
    # If pressed twice within 10 seconds -> Exit the script completely
    if current_time - INTERRUPT_STATE["last_press_time"] < 10.0:
        print("\n\n🛑 Double Ctrl+C detected! Exiting safely...")
        sys.exit(0)
        
    # Otherwise, set the skip flag and warn the user
    INTERRUPT_STATE["last_press_time"] = current_time
    INTERRUPT_STATE["skip_cooldown"] = True
    print("\n\n⚠️  [Interrupt] Press Ctrl+C again within 10 seconds to EXIT.")

# Intercept OS-level Ctrl+C signals
signal.signal(signal.SIGINT, handle_sigint)

# ── Configuration & Data Management ────────────────────────────────────────────

QUICK_TEST_MODE  = False
RESUME_FROM_SAVE = True  
TARGET_SAVE_DIR  = "test_run_2026_06_18_045952"   

population_cap  = 100 if not QUICK_TEST_MODE else 6

# ── Logical Reasoning Training Dataset ─────────────────────────────────────────
TRAINING_DATASET = [
    # Algebra / Composition
    {"input": "If f(x) = 3x^2 - 2x + 1, what is the value of f(f(2))? Provide only the number.", "output": "226"},
    
    # Spatial Reasoning
    {"input": "A solid 3x3x3 wooden cube is painted red on all its outside faces. It is then cut into 27 smaller 1x1x1 cubes. How many of these smaller cubes have exactly two red faces? Provide only the number.", "output": "12"},
    
    # Combinatorics
    {"input": "How many unique 4-letter permutations can be made from the letters in the word 'BOOK'? Provide only the number.", "output": "12"},
    
    # Multi-variable Word Problem
    {"input": "Alice is twice as old as Bob. In 5 years, the sum of their ages will be 46. How old is Bob right now? Provide only the number.", "output": "12"},
    
    # Algorithmic Logic / Cryptography
    {"input": "In a Caesar cipher shifted forward by 3 letters (where A becomes D), how do you write the word 'SYSTEM'? Provide only the 6-letter capitalized word.", "output": "VBVWHP"},
    
    # Data Structures
    {"input": "A binary tree has a root node with value 10. Its left child is 5 and its right child is 15. The left child of 5 is 2. The right child of 15 is 20. What is the sum of all leaf nodes in this tree? Provide only the number.", "output": "22"},
    
    # Physics / Kinematics
    {"input": "A train travels at 60 mph for 2 hours, then speeds up to 90 mph for 1.5 hours. What is its average speed for the entire journey in mph? Provide only the number, rounded to the nearest whole number.", "output": "73"},
    
    # Boolean Logic
    {"input": "Evaluate the following boolean logic statement, assuming A=True, B=False, C=True: (A AND NOT B) OR (B AND C). Provide only True or False.", "output": "True"},
    
    # Multi-constraint Reasoning
    {
        "input": "Write a Python function to calculate the Fibonacci sequence. Requirements: 1) Must use recursion. 2) Must include docstrings. 3) Output ONLY the raw Python code, no markdown code blocks or explanations.",
        "output": "A valid recursive Python function calculating Fibonacci, containing a docstring, with zero markdown backticks and zero conversational text."
    },
    {
        "input": "A farmer has a rectangular field that is 100 meters long and 50 meters wide. He wants to plant trees every 10 meters along the perimeter. How many trees does he need? Requirements: 1) Think step-by-step. 2) The final sentence must be exactly: 'The total number of trees is X.'",
        "output": "The AI should calculate the perimeter (300m), divide by 10 (30 trees), provide step-by-step reasoning, and end with the exact specified string 'The total number of trees is 30.'"
    },
    {
        "input": "Summarize the plot of the movie 'The Matrix' in exactly three sentences. Requirements: 1) You must write from the perspective of Agent Smith. 2) You must use the word 'inevitable'.",
        "output": "A three-sentence summary of The Matrix. Written with a hostile, robotic tone (Agent Smith persona). The word 'inevitable' is present."
    },
    {
        "input": "Convert the following JSON object into an XML string: {\"user\": {\"id\": 42, \"name\": \"Alice\", \"roles\": [\"admin\", \"editor\"]}}. Requirements: 1) The roles must be nested as individual <role> tags inside a <roles> parent block. 2) Output nothing but the XML.",
        "output": "Properly nested XML representing the JSON data. <roles><role>admin</role><role>editor</role></roles>. No pleasantries or markdown blocks."
    },
    {
        "input": "Solve this logic puzzle: There are 3 boxes. Box A says 'Gold is here'. Box B says 'Gold is not here'. Box C says 'Gold is not in Box A'. Only one box is telling the truth. Which box has the gold? Requirements: 1) Briefly explain the logical deduction. 2) Conclude with a JSON object: {\"gold_location\": \"Box X\"}",
        "output": "Logical deduction showing that if A is true, C is false, but if B is false gold is in B... wait, the logic leads to Gold being in Box B. The output must end with {\"gold_location\": \"Box B\"}."
    }
]


gdm = GeneralDataManager(run_dir=TARGET_SAVE_DIR, resume_from_save=RESUME_FROM_SAVE)

# ── Logging Setup (Clean UI) ───────────────────────────────────────────────────
log_file_path = os.path.join(gdm.run_dir, "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file_path)] # Only write verbose to file, keep console clean
)
logger = logging.getLogger(__name__)
logger.info(f"Initialized Run Directory: {gdm.run_dir}")

print(f"\n📁 Run Directory: {gdm.run_dir}")
print(f"⚙️  Ollama Threads Limited to: {os.environ['OLLAMA_NUM_THREADS']}")

runner = PromptChainRunner(verbose=False)
runner.update_model_registry(benchmark=False)
available_models = list(runner.model_registry.keys())

def get_chain_id(chain: list) -> str:
    return f"chain_{hashlib.md5(str(chain).encode('utf-8')).hexdigest()[:12]}"


# ── Core Execution & Evaluation Logic ──────────────────────────────────────────

def evaluate_population_with_cache(population_records: list[dict], generation_num: int, task: dict) -> list[tuple]:
    evaluated_data = []
    final_records = []
    
    initial_input = task["input"]
    solution_output = task["output"]
    task_id = hashlib.md5(initial_input.encode('utf-8')).hexdigest()[:8]
    
    heritage_db = gdm.heritage_data_manager.local_heritage_database.get("prompt_chains", {})
    to_execute = []

    for record in population_records:
        chain_id = record["chain_id"]
        chain = record["chain"]
        parents = record.get("parents", [])
        metadata = record.get("metadata", {})
        metadata["generation"] = generation_num

        heritage_record = heritage_db.get(chain_id, {})
        cached_fitness = heritage_record.get("fitness")
        cached_task = heritage_record.get("metadata", {}).get("task_id")
        
        if cached_fitness is not None and cached_task == task_id:
            existing_meta = heritage_record.get("metadata", {})
            existing_meta["generation"] = generation_num
            final_records.append((chain_id, chain, cached_fitness, existing_meta))
            logger.info(f"  [Cache Hit] {chain_id}")
            
            # 🚨 FIX: Update heritage history only, do NOT overwrite the population file!
            gdm.id_to_promptchain_manager.add_or_update(chain_id, chain)
            gdm.heritage_data_manager.update_population([(chain_id, chain, cached_fitness, existing_meta)])
        else:
            to_execute.append((chain_id, chain, parents, metadata))

    total = len(to_execute)
    if total > 0:
        print(f"    Evaluating {total} novel chains for this task...")
        start_time = time.time()
        
        for i, (chain_id, chain, parents, metadata) in enumerate(to_execute):
            sys.stdout.write(f"\r    ⏳ Progress: [{i}/{total}] | Running: {chain_id}...")
            sys.stdout.flush()
            
            # 🚨 FIX: Pass 'runner' correctly and DO NOT double-run the chain!
            fitness, telemetry = fitness_algorithm.evaluate_prompt_chain(chain, runner, initial_input, solution_output)
            
            metadata.update(telemetry) 
            metadata["creation_time"] = datetime.now(timezone.utc).isoformat()
            metadata["task_id"] = task_id  
            
            safe_chain_parts = []
            for step in chain:
                if isinstance(step, (list, tuple)) and len(step) >= 2:
                    segs = step[1] if isinstance(step[1], list) else [str(step[1])]
                    safe_chain_parts.append(f"[{step[0]}: '{''.join(segs)}']")
                elif isinstance(step, (list, tuple)) and len(step) == 1:
                    safe_chain_parts.append(f"[{step[0]}: '']")
                else:
                    safe_chain_parts.append(f"[Malformed Mutant]")
                    
            chain_str = " -> ".join(safe_chain_parts)
            
            # 🚨 FIX: Safely pull the output from telemetry for logging
            final_output = telemetry.get("final_output", "Error: Malformed Output")
            log_out = final_output.replace('\n', ' ')[:80] + ('...' if len(final_output) > 80 else '')
            
            final_records.append((chain_id, chain, fitness, metadata))
            
            logger.info(f"  [Scored] {chain_id} | Fit: {fitness:.4f} | Time: {telemetry.get('time_taken', 0.0):.2f}s")
            logger.info(f"           Chain:    {chain_str[:120]}{'...' if len(chain_str) > 120 else ''}")
            logger.info(f"           Expected: '{solution_output}'")
            logger.info(f"           Output:   '{log_out}'")
            
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r    ✅ Progress: [{total}/{total}] | Completed in {elapsed:.1f}s                 \n")
        sys.stdout.flush()
            
    # 🚨 FIX: Update heritage history only, do NOT overwrite the population file!
    for rec in final_records:
        gdm.id_to_promptchain_manager.add_or_update(rec[0], rec[1])
    gdm.heritage_data_manager.update_population(final_records)
    
    return final_records

# ── Load Gene Pool Early for Mutators ──────────────────────────────────────────
gene_pool_manager = GenePoolManager()
segments = gene_pool_manager.load_prompt_segments()
if not segments:
    gene_pool_manager.run_pipeline()
    segments = gene_pool_manager.load_prompt_segments()

# ── GA operator instances (Silenced for UI) ────────────────────────────────────

fitness_algorithm = FitnessCalculation()

selection_algorithm = PromptChainSelection(
    selection_algorithm=StochasticUniversalSampling(selection_ratio=0.5),
    verbose=False
)

recombination_algorithm = PromptChainRecombination(
    prompt_chain_pairing_instance=LineagePairing(verbose=False),
    prompt_chain_crossover_instance=LineageCrossover(verbose=False),
    prompt_crossover_instance=ModelBasedCrossover(verbose=False),
    gdm=gdm
)

mutation_algorithm = PromptChainMutation(
    base_mutation_chance=0.1, 
    gdm=gdm,
    verbose=False,
    mutation_methods=[
        SemanticLLMMutation(runner=runner, mutator_model="qwen2.5-coder:0.5b", verbose=False).mutate,
        DeleteMutation(min_segment_fraction=0.1, max_segment_fraction=0.3).mutate,
        GenePoolSearchMutation(segments=segments, verbose=False).mutate 
    ]
)

migration_algorithm = PromptChainMigration(base_migration_chance=0.20,gdm=gdm, verbose=False) 

sa_algorithm = SimulatedAnnealing(
    evaluator_func=evaluate_population_with_cache,
    micro_mutator=MicroMutation(runner=runner, mutator_model="qwen2.5-coder:0.5b", verbose=False),
    gdm=gdm, steps=3, initial_temp=10.0, elite_selection_ratio=0.2, verbose=False
)

replacement_algorithm = PromptChainReplacement(
    replacement_algorithm=LineageReplacement(population_cap=population_cap, verbose=False),
    gdm=gdm 
)


# ── Population Initialization / Resumption ─────────────────────────────────────

current_population_records = []
start_generation = 0
saved_pop_data = gdm.population_data_manager.local_population_data

if RESUME_FROM_SAVE and saved_pop_data and saved_pop_data.get("population"):
    saved_pop = saved_pop_data["population"]
    
    max_gen = 0
    for entry in saved_pop:
        chain_id = entry[0]
        chain_tuples = [tuple(step) for step in entry[1]]
        fitness = entry[2]
        metadata = entry[3]
        current_population_records.append((chain_id, chain_tuples, fitness, metadata))
        
        # 🚨 FIX: Dynamically find the highest generation among all survivors
        gen_val = metadata.get("generation", 0)
        if isinstance(gen_val, list):
            gen_val = max(gen_val) if gen_val else 0
        if gen_val > max_gen:
            max_gen = gen_val
            
    start_generation = max_gen + 1
    print(f"\n💾 Resuming execution safely at Generation {start_generation}.")

# ── THE FIX: HERITAGE RECONSTRUCTION ──
elif RESUME_FROM_SAVE and gdm.heritage_data_manager.local_heritage_database.get("prompt_chains"):
    print(f"\n🩹 Active population missing. Reconstructing from Heritage Database...")
    heritage_db = gdm.heritage_data_manager.local_heritage_database.get("prompt_chains", {})
    
    # 1. Find the highest generation achieved
    max_gen = 0
    for chain_id, data in heritage_db.items():
        gens = data.get("generation", [])
        if gens and max(gens) > max_gen:
            max_gen = max(gens)
            
    # 2. Resurrect all individuals from that generation
    for chain_id, data in heritage_db.items():
        if max_gen in data.get("generation", []):
            chain = gdm.id_to_promptchain_manager.get_promptchain_from_id(chain_id)
            if chain:
                chain_tuples = [tuple(step) for step in chain]
                fitness = data.get("fitness")
                metadata = data.get("metadata", {})
                current_population_records.append((chain_id, chain_tuples, fitness, metadata))
                
    start_generation = max_gen + 1
    print(f"🚀 Reconstructed {len(current_population_records)} individuals from Gen {max_gen}!")
    print(f"🚀 Resuming execution at Generation {start_generation}.")
    
    # Save it back to the active population file to fix the state
    gdm.sync_population(current_population_records)

else:
    print("\n🌱 Generating initial population...")
    gene_pool_manager = GenePoolManager()
    segments = gene_pool_manager.load_prompt_segments()
    if not segments:
        gene_pool_manager.run_pipeline()
        segments = gene_pool_manager.load_prompt_segments()

    initial_pop_generator = InitialPopulationGenerator(
        segments=segments, available_models=available_models, population_cap=population_cap,
        min_chain_length=1, max_chain_length=1 
    )
    
    raw_chains = initial_pop_generator.generate()
    initial_dicts = []
    for chain in raw_chains:
        chain_id = get_chain_id(chain)
        metadata = {"prefix_len": 1, "recombination_mode": "initial"}
        gdm.register_intermediary_chain(chain_id, chain, [], metadata)
        initial_dicts.append({"chain_id": chain_id, "chain": chain, "parents": [], "metadata": metadata})
        
    print("\n🧬 Evaluating Initial Population (Generation 0)...")
    init_task = random.choice(TRAINING_DATASET)
    print(f"    Task: {init_task['input']}")
    current_population_records = evaluate_population_with_cache(initial_dicts, generation_num=0, task=init_task)
    start_generation = 1


# ── GA main loop ───────────────────────────────────────────────────────────────

try:
    gen = start_generation
    while True: # Indefinite Training Loop
        print(f"\n{'='*60}\n=== GENERATION {gen} ===\n{'='*60}")
        gen_start_time = time.time()
        
        # 0. Assign Task
        current_task = random.choice(TRAINING_DATASET)
        print(f"🎯 Current Task: {current_task['input']}")
        print(f"👥 Population Size: {len(current_population_records)}")

        # 1. Operators
        selected_records = selection_algorithm.select_prompt_chains(current_population_records)
        offspring_dicts = recombination_algorithm.recombine_prompt_chains(selected_records)
        
        # Adjust dynamic rates based on parent performance
        mutation_algorithm.adjust_mutation_rate(current_population_records)
        migration_algorithm.adjust_migration_rate(current_population_records)
        
        # Apply operators
        offspring_dicts = mutation_algorithm.mutate_population(offspring_dicts)
        offspring_dicts = migration_algorithm.migrate_population(offspring_dicts)

        # 2. Evaluate
        print(f"\n🧪 Evaluating Offspring...")
        evaluated_offspring = evaluate_population_with_cache(offspring_dicts, generation_num=gen, task=current_task)

        # 3. Simulated Annealing
        print(f"🔬 Annealing Elites...")
        evaluated_offspring = sa_algorithm.process_population(evaluated_offspring, generation_num=gen, task=current_task)

        # 4. Replacement
        current_population_records = replacement_algorithm.replace_population(
            current_population_records, evaluated_offspring
        )

        gdm.sync_population(current_population_records)

        gen_elapsed = time.time() - gen_start_time
        print(f"✅ Generation {gen} Complete in {gen_elapsed:.1f}s")

        gen += 1
        
        # ── Interruptible Cooldown Timer ──
        cooldown_seconds = 180
        INTERRUPT_STATE["skip_cooldown"] = False # Reset flag for the new generation
        
        print(f"\n❄️  Cooling down for {cooldown_seconds//60} minutes to clear VRAM...")
        
        for remaining in range(cooldown_seconds, 0, -1):
            # Check the global flag every second
            if INTERRUPT_STATE["skip_cooldown"]:
                print("\n    ⏭️  Cooldown skipped by user! Initiating next generation...")
                break
                
            sys.stdout.write(f"\r    ⏳ Resuming in {remaining} seconds... ")
            sys.stdout.flush()
            time.sleep(1)
            
        # If we finished the loop naturally without skipping
        if not INTERRUPT_STATE["skip_cooldown"]:
            sys.stdout.write("\r    ✅ Cooldown complete!                      \n")
            sys.stdout.flush()
        

except KeyboardInterrupt:
    print("\n\n[!] KeyboardInterrupt detected (Ctrl+C).")
    print("[!] Saving the last stable generation state to database...")
    gdm.sync_population(current_population_records)
    print(f"[!] Saved successfully to {gdm.run_dir}. Exiting.")
    sys.exit(0)