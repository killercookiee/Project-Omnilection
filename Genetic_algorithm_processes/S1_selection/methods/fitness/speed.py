# Input time taken (float) for example in seconds
# Output a normalized speed score between 0 and 1
# The function is a smoothstep function between (0, 1) and (speed_limit, 0)

# Customizable parameters:
# speed limit: maximum time taken for score to be non zero (float)          | default is 10 seconds

import time
class Speed:
    def __init__(self, speed_limit=10.0):
        self.speed_limit = speed_limit

    def calculate_speed_score(self, input_time):
        """
        Speed function to normalize time taken into a score between 0 and 1.
        
        Parameters:
        - input_time: The time taken (float)
        - speed_limit: Maximum time for non-zero score (float)
        
        Returns:
        - speed_score: The normalized speed score (float)
        """
        if input_time >= self.speed_limit:
            return 0.0
        else:
            # Smoothstep function
            x = input_time / self.speed_limit
            return 1 - (3 * x**2 - 2 * x**3)  # Smoothstep formula
    

if __name__ == "__main__":
    speed_instance = Speed(speed_limit=10.0)

    # Simulate timing a process
    start_time = time.time()
    time.sleep(3)  # Simulate a process taking 3 seconds
    end_time = time.time()
    elapsed_time = end_time - start_time

    speed_score = speed_instance.calculate_speed_score(elapsed_time)
    print(f"Elapsed Time: {elapsed_time} seconds")
    print(f"Speed Score: {speed_score}")