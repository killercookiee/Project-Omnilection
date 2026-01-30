# Input parents and offspring prompt chains
# Output offspring prompt chains after recombination

# for each prompt in the prompt chain of the offspring, pair with all prompts with the same model in the parents' prompt chains
# undergo crossover among the paired prompts to create new prompt segments

# Customizable parameters:
# - Crossover method (e.g., single-point, multi-point, uniform)


from .N_crossover_multiparent import NCrossoverMultiparent


class TotalParentCrossover:
    def __init__(self, crossover_method=None):
        self.crossover_method = crossover_method if crossover_method is not None else NCrossoverMultiparent(crossover_num=2)
    
    def select_prompts_by_model(self, parents, model):
        selected_prompts = []
        for parent in parents:
            for prompt in parent:
                if prompt[0] == model:
                    selected_prompts.append(prompt)
        return selected_prompts

    def crossover(self, parents, offspring):
        for i in range(len(offspring)):
            model = offspring[i][0]
            selected_prompts = self.select_prompts_by_model(parents, model)
            if len(selected_prompts) > 1:
                new_prompt = self.crossover_method.crossover(selected_prompts)
                offspring[i] = new_prompt
        return offspring
    

if __name__ == "__main__":
    parents = [
        [("gpt-3.5-turbo", "Prompt A1"), ("gpt-4", "Prompt B1")],
        [("gpt-3.5-turbo", "Prompt A2"), ("gpt-4", "Prompt B2")]
    ]
    offspring = [("gpt-3.5-turbo", "Initial Prompt A"), ("gpt-4", "Initial Prompt B")]
    
    model_random_crossover = TotalParentCrossover()
    new_offspring = model_random_crossover.crossover(parents, offspring)
    print(f"New Offspring: {new_offspring}")