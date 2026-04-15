"""
Data/general_datamanager.py
"""

# Example heritage data structure (in heritage_data.json):
# heritage_database = {
#     "generations": 5,
#     "version": "1.0",
#     "size": 2,
#     "prompt_chains": {
#         "prompt_chain_id_3": {
#             "parents": ["parent_prompt_chain_id_1", "parent_prompt_chain_id_2"],
#             "fitness": 0.8,
#             "generation": [1, 2, 3, 4, 5],
#             "metadata": {
#                 "creation_time": "2024-01-01T12:00:00Z",
#                 "modification_time": "2024-01-01T12:00:00Z",
#                 "other_info": "..."
#             }
#         },
#         "prompt_chain_id_4": {
#             "parents": ["parent_prompt_chain_id_3"],
#             "fitness": 0.6,
#             "generation": [4, 5],
#             "metadata": {
#                 "creation_time": "2024-01-02T12:00:00Z",
#                 "modification_time": "2024-01-02T12:00:00Z",
#                 "other_info": "..."
#             }
#         },
#         ...
#     }
# }


# Example population data structure (in population_data.json):
# population_data = {
#     "population": [
#         (prompt_chain_id_1, [("modelA", ["prompt_section_1", "prompt_section_2"], ("modelB", ["prompt_section_3", "prompt_section_4"]), ...], fitness_score_1, metadata_1),  # prompt chain 1
#         (prompt_chain_id_2, [("modelC", ["prompt_section_5", "prompt_section_6"], ("modelD", ["prompt_section_7", "prompt_section_8"]), ...], fitness_score_2, metadata_2),  # prompt chain 2
#         ...
#     ],
#     "metadata": {
#         "generation": 5,
#         "creation_time": "2024-01-01T12:00:00Z",
#         "modification_time": "2024-01-01T12:00:00Z",
#         "other_info": "..."
#     }
# } 


# Example id_to_promptchain data structure:
# id, prompt_chain
# prompt_chain_id_1, [("modelA", ["prompt_section_1", "prompt_section_2"]), ("modelB", ["prompt_section_3", "prompt_section_4"]), ...]
# prompt_chain_id_2, ...



import json


class HeritageDataManager:
    def __init__(self, heritage_database_file: str = "heritage_data.json"):
        self.heritage_database_file = heritage_database_file
        self.local_heritage_database = self._load_heritage_database()

    def _load_heritage_database(self):
        # Load the heritage database from the file, or initialize an empty database if the file doesn't exist
        try:
            with open(self.heritage_database_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        
    def _save_heritage_database(self):
        # Save the heritage database to the file
        with open(self.heritage_database_file, "w") as f:
            json.dump(self.local_heritage_database, f, indent=4)

    def _get_content_from_ID(self, prompt_chain_id: str):
        # Retrieve the content of a prompt chain from the id_to_promptchain.json file
        pass
        
        
    def add_new(self,
                prompt_chain_id: str,
                parents: list[str],
                fitness: float,
                generation: list[int],
                metadata: dict = None
    ):
        # Add a new prompt chain to the heritage database
        pass

    def update_population(self,
                          current_population: list[list[tuple]],
    ):
        # Update the heritage database with the current population, modifying existing prompt chain generation and adding new ones as needed
        pass

    def calculate_lineage_score(self, prompt_chain_id: str, index) -> float:
        # Calculate the lineage score for a given prompt chain
        pass


class PopulationDataManager:
    def __init__(self):
        self.local_population_data = None

    def _load_population_data(self, file_path):
        pass

    def _save_population_data(self, file_path):
        pass

    def update_population_data(self, current_population):
        pass


class IDToPromptChainManager:
    def __init__(self):
        pass

    def _load_id_to_promptchain_data(self, file_path):
        pass

    def _save_id_to_promptchain_data(self, file_path):
        pass

    def get_promptchain_from_id(self, prompt_chain_id):
        pass


class GeneralDataManager:
    def __init__(self):
        self.heritage_data_manager = HeritageDataManager()
        self.population_data_manager = PopulationDataManager()
        self.id_to_promptchain_manager = IDToPromptChainManager()