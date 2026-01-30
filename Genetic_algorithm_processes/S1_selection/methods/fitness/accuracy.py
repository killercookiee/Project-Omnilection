# Input a accuracy score (percentage between 0 and 100)
# Output a normalized accuracy score between 0 and 1
# The function is a exponential scaling function from (0,0) to (100, 1)

# Customizable parameters:
# scaling factor: controls the steepness of the curve (float)               | default is 0.05


class Accuracy:
    def __init__(self, scaling_factor=0.05, testing_model = None):
        self.scaling_factor = scaling_factor
        self.testing_model = testing_model

    def get_accuracy(self, prompt_output, testing_model=None): # Place holder for actual accuracy computation
        """Receives prompt output and testing_model to compute accuracy score percentage between 0 and 100."""
        # Placeholder implementation - in practice, this would compare prompt_output to a reference answer
        # prompt_accuracy = self.testing_model.evaluate(prompt_output)
        prompt_accuracy = 100
        return prompt_accuracy
    
    def normalize_accuracy_score(self, accuracy_value=None):
        """
        Accuracy function to normalize accuracy scores between 0 and 1 using a function
        
        Parameters:
        - input_score: The accuracy score (float between 0 and 100)
        - scaling_factor: Controls the steepness of the curve (float)
        
        Returns:
        - normalized_score: The normalized accuracy score (float)
        """
        if accuracy_value is None:
            accuracy_value = self.prompt_accuracy
        if accuracy_value < 0:
            return 0.0
        elif accuracy_value > 100:
            return 1.0
        else:
            import math
            normalized_score = 1 - math.exp(-self.scaling_factor * accuracy_value)
            return normalized_score
        
    def get_accuracy_score(self, prompt_output):
        accuracy_value = self.get_accuracy(prompt_output)
        accuracy_score = self.normalize_accuracy_score(accuracy_value)
        return accuracy_score
        
if __name__ == "__main__":
    accuracy_instance = Accuracy(scaling_factor=0.05)

    prompt_output = "The capital of France is Paris."
    accuracy_score = accuracy_instance.get_accuracy_score(prompt_output)
    print(f"Computed Accuracy Score for '{prompt_output}': {accuracy_score}")