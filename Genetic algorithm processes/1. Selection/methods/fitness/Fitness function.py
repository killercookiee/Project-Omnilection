# Input of scores from accuracy, limit, model cost, token limit, speed functions
# Combine these scores into a single fitness score for selection

# Fitness = (alpha * accuracy_score + beta * model_cost_score) * limit(gamma * token_limit_score) * limit( delta * speed_score)

# Customizable parameters:
# alpha, beta, gamma, delta: weights for each component in the fitness calculation         | default is 1.0 for all


from accuracy import accuracy
from model_cost import model_cost
from limit import limit
from speed import speed
from token_limit import token_limit

class FitnessFunction:
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, accuracy_func=accuracy, model_cost_func=model_cost, limit_func=limit, speed_func=speed, token_limit_func=token_limit):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Functions
        self.accuracy_func = accuracy_func
        self.model_cost_func = model_cost_func
        self.limit_func = limit_func
        self.speed_func = speed_func
        self.token_limit_func = token_limit_func

    def compute_fitness(self, accuracy_value, model_cost_value, token_limit_value, speed_value):
        acc_score = self.accuracy_func(accuracy_value)
        cost_score = self.model_cost_func(model_cost_value)
        limit_score = self.limit_func(self.gamma * self.token_limit_func(token_limit_value))
        speed_value = self.limit_func(self.delta * self.speed_func(speed_value))

        fitness = (self.alpha * acc_score + self.beta * cost_score) * limit_score * speed_value
        return fitness