# Input a list of prompt_chains (d-dimensional list), 
# Output a list of index d-dimensional list representing each row with d-dimension, meaning successful pairing


class IndexPairing:
    def __init__(self, stationary_indices=None):
        self.stationary_indices = stationary_indices if stationary_indices is not None else [0]

    def index_pairing(self,data):
        """
        Create index pairings for a d-dimensional list.
        
        Parameters:
        - data: A list of d lists, each containing values for that dimension (list of lists)
        
        Returns:
        - index_pairs: A list of lists, where each inner list contains indices representing a row (list of lists)
        """
        if not data or not all(isinstance(col, list) for col in data):
            return []
        
        num_rows = len(data[0])
        for col in data:
            if len(col) != num_rows:
                raise ValueError("All columns must have the same number of rows.")
        
        index_pairs = []
        for i in range(num_rows):
            row_indices = [i if j not in self.stationary_indices else self.stationary_indices[j] for j in range(len(data))]
            index_pairs.append(row_indices)
        
        return index_pairs


if __name__ == "__main__":
    prompt_chain_population = [
        [{"gpt-3.5-turbo", "Prompt 1"}, {"gpt-4", "Prompt 2"}],
        [{"gpt-4", "Prompt 3"}, {"gpt-3.5-turbo", "Prompt 4"}],
        [{"gpt-3.5-turbo", "Prompt 5"}, {"gpt-4", "Prompt 6"}],
        [{"gpt-4", "Prompt 7"}, {"gpt-3.5-turbo", "Prompt 8"}]
    ]

    index_pairing = IndexPairing(stationary_indices=[0])