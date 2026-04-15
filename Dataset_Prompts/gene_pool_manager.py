"""
Dataset_Prompts/gene_pool_manager.py
"""


# Create prompt_segments.db and initial_population.db for the Genetic Algorithm.

# This script:
# 1. Downloads prompt dataset from Hugging Face and Github repositories locally
# 2. Extract prompt from each database
# 3. Split prompts into sections and save them in prompt_segments (also keep the original prompt in gene_pool)

# gene_pool.db format:
# ...

# Current available datasets:
# 1. linexjilin:GPTs - https://github.com/linexjlin/GPTs.git
# 2. f:prompts.chat - https://github.com/f/awesome-chatgpt-prompts.git
# 3. x1xhlol:system-prompts-and-models-of-ai-tools - https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools.git

import os
import re
import subprocess
import glob
import csv
import yaml
import urllib.request

class GenePoolManager:
    def __init__(self):
        # Updated to specify the exact download strategy for each source
        self.datasets = {
            "GPTs": {
                "strategy": "git_sparse",
                "url": "https://github.com/linexjlin/GPTs.git",
                "paths": ["prompts/"]
            },
            "prompts_chat": {
                "strategy": "raw_file",
                # This points to the raw text version of the CSV on GitHub
                "url": "https://raw.githubusercontent.com/f/prompts.chat/main/prompts.csv",
                "filename": "prompts.csv"
            },
            "system_prompts": {
                "strategy": "git_shallow",
                "url": "https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools.git"
            }
        }
        self.data_dir = "./Dataset_Prompts/datasets"
        self.gene_pool_file = "./Dataset_Prompts/gene_pool.yaml"
        self.segments_file = "./Dataset_Prompts/prompt_segments.yaml"

        self.gene_pool = []
        self.segments = []
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _download_datasets(self):
        print("Downloading datasets...")
        for name, data in self.datasets.items():
            repo_path = os.path.join(self.data_dir, name)
            strategy = data.get("strategy")

            # --- Handle Raw File Downloads ---
            if strategy == "raw_file":
                if not os.path.exists(repo_path):
                    os.makedirs(repo_path)
                
                file_path = os.path.join(repo_path, data["filename"])
                if not os.path.exists(file_path):
                    print(f"Downloading raw file for {name}...")
                    urllib.request.urlretrieve(data["url"], file_path)
                else:
                    print(f"Dataset {name} already exists. Skipping download.")
                continue # Move to the next dataset

            # --- Handle Git Downloads ---
            if os.path.exists(repo_path):
                print(f"Dataset {name} already exists. Skipping clone.")
                continue

            url = data["url"]

            if strategy == "git_sparse":
                print(f"Sparse cloning {name} (Only grabbing specified folders)...")
                subprocess.run([
                    "git", "clone", "--filter=blob:none", "--sparse", "--depth", "1", url, repo_path
                ], check=True)
                
                # Notice I added --skip-checks here just in case, though it's grabbing a folder now
                subprocess.run(
                    ["git", "-C", repo_path, "sparse-checkout", "set", "--skip-checks"] + data["paths"], 
                    check=True
                )
                
            elif strategy == "git_shallow":
                print(f"Fast shallow cloning {name} (Grabbing all files)...")
                subprocess.run([
                    "git", "clone", "--depth", "1", "--single-branch", url, repo_path
                ], check=True)

    def _extract_prompts(self):
        print("Extracting prompts from local repositories...")
        
        # 1. linexjlin/GPTs (Markdown files in the /prompts directory)
        gpts_path = os.path.join(self.data_dir, "GPTs", "prompts", "*.md")
        for filepath in glob.glob(gpts_path):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find everything between ```markdown and ```
                # re.DOTALL ensures it captures multiple lines of text
                # re.IGNORECASE handles if they typed ```Markdown
                matches = re.findall(r'```markdown\s*(.*?)\s*```', content, flags=re.DOTALL | re.IGNORECASE)
                
                if matches:
                    # Some files might have multiple markdown blocks, so we add all of them
                    for match in matches:
                        self.gene_pool.append(match.strip())
                else:
                    # If a file is missing the ```markdown block entirely, 
                    # we skip it to keep the data pool clean.
                    pass

        # 2. f/awesome-chatgpt-prompts (Extracting from prompts.csv)
        csv_path = os.path.join(self.data_dir, "prompts_chat", "prompts.csv")
        if os.path.exists(csv_path):
            
            # --- ADD THIS LINE HERE ---
            csv.field_size_limit(10000000) # Increases limit to 10 million characters per cell
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'prompt' in row:
                        self.gene_pool.append(row['prompt'].strip())

        # 3. x1xhlol/system-prompts (Text files scattered throughout the repo)
        sys_prompts_path = os.path.join(self.data_dir, "system_prompts", "**", "*.txt")
        for filepath in glob.glob(sys_prompts_path, recursive=True):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.gene_pool.append(f.read().strip())

    def _split_prompts_into_sections(self):
        print("Splitting prompts into distinct segments...")
        for prompt in self.gene_pool:
            # Splitting by double newline to separate logical blocks/paragraphs
            sections = [sec.strip() for sec in prompt.split('\n\n') if sec.strip()]
            self.segments.extend(sections)

    def _clean_up_genepool(self):
        print("Cleaning up the gene pool...")
        # Remove duplicates by converting to a set, then back to a list
        unique_segments = list(set(self.segments))
        
        # Remove empty or extremely short/broken segments (less than 15 characters)
        self.segments = [seg for seg in unique_segments if len(seg) > 15]

    def _save_prompt_segments(self):
        print("Saving readable YAML files...")
        
        # This forces the PyYAML library to use the `|` block style for clean formatting
        def literal_str_presenter(dumper, data):
            if '\n' in data or '"' in data or "'" in data or len(data) > 80:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)
            
        yaml.add_representer(str, literal_str_presenter)

        # Save Original Gene Pool
        with open(self.gene_pool_file, 'w', encoding='utf-8') as f:
            # allow_unicode=True is CRITICAL for exotic characters
            yaml.dump(self.gene_pool, f, allow_unicode=True, sort_keys=False)

        # Save Segments
        with open(self.segments_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.segments, f, allow_unicode=True, sort_keys=False)

    def load_prompt_segments(self) -> list:
        if not os.path.exists(self.segments_file):
            print("No segments file found. Run pipeline first.")
            return []
            
        print("Loading YAML segments into memory...")
        with open(self.segments_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or []

    def load_gene_pool(self) -> list:
        if not os.path.exists(self.gene_pool_file):
            print("No gene pool file found. Run pipeline first.")
            return []
            
        print("Loading YAML gene pool into memory...")
        with open(self.gene_pool_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or []

    def run_pipeline(self):
        """Helper method to run the entire process end-to-end."""
        self._download_datasets()
        self._extract_prompts()
        self._split_prompts_into_sections()
        self._clean_up_genepool()
        self._save_prompt_segments()
        print("Pipeline complete!")


# Usage example:
if __name__ == "__main__":
    manager = GenePoolManager()
    manager.run_pipeline()
    
    # Test loading it back into memory
    gene_pool = manager.load_gene_pool()
    print(f"Successfully loaded {len(gene_pool)} prompts for the Genetic Algorithm!")
    segments = manager.load_prompt_segments()
    print(f"\nSuccessfully loaded {len(segments)} clean segments for the Genetic Algorithm!")