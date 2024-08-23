import numpy as np
import random

def grid_search(traces, batch_size, evaluation_fn):
    best_x, best_y = None, None
    best_score = -np.inf
    
    for x in range(1, batch_size):
        for y in range(1, batch_size):
            score = evaluation_fn(traces, x, y, batch_size=batch_size)
            if score > best_score:
                best_x, best_y = x, y
                best_score = score
    
    return best_x, best_y, best_score

def hill_climbing_search(traces, batch_size, evaluation_fn, max_iters=50):
    np.random.seed(42)
    # Random starting point
    current_x = np.random.randint(1, batch_size)
    current_y = np.random.randint(1, batch_size)
    current_score = evaluation_fn(traces, current_x, current_y, batch_size=batch_size)
    
    for _ in range(max_iters):
        neighbors = [
            (current_x - 1, current_y),
            (current_x + 1, current_y),
            (current_x, current_y - 1),
            (current_x, current_y + 1)
        ]
        neighbors = [(x, y) for x, y in neighbors if 1 <= x < batch_size and 1 <= y < batch_size]
        
        next_x, next_y = None, None
        next_score = current_score
        
        for x, y in neighbors:
            score = evaluation_fn(traces, x, y, batch_size=batch_size)
            if score > next_score:
                next_x, next_y = x, y
                next_score = score
        
        if next_x is None or next_y is None:
            break  # No better neighbor found
        
        current_x, current_y = next_x, next_y
        current_score = next_score
    
    return current_x, current_y, current_score

def simulated_annealing_search(traces, batch_size, evaluate_formula, initial_temp=100, cooling_rate=0.95, max_iters=1000):
    # Random starting point
    current_x = np.random.randint(1, batch_size)
    current_y = np.random.randint(1, batch_size)
    current_score = evaluate_formula(traces, current_x, current_y, batch_size=batch_size)
    
    best_x, best_y = current_x, current_y
    best_score = current_score
    
    temperature = initial_temp
    
    for _ in range(max_iters):
        # Generate a random neighboring solution
        next_x = current_x + random.choice([-1, 1])
        next_y = current_y + random.choice([-1, 1])
        
        # Ensure the next point is within bounds
        next_x = np.clip(next_x, 1, batch_size - 1)
        next_y = np.clip(next_y, 1, batch_size - 1)
        
        next_score = evaluate_formula(traces, next_x, next_y)
        
        # Calculate the change in score
        delta_score = next_score - current_score
        
        # Determine whether to move to the next point
        if delta_score > 0 or np.exp(delta_score / temperature) > np.random.rand():
            current_x, current_y = next_x, next_y
            current_score = next_score
            
            # Update best found solution
            if current_score > best_score:
                best_x, best_y = current_x, current_y
                best_score = current_score
        
        # Cool down the temperature
        temperature *= cooling_rate
        
        # Stop if temperature is too low
        if temperature < 1e-3:
            break
    
    return best_x, best_y, best_score

def particle_swarm_optimization(traces, batch_size, evaluation_fn, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    # Initialize the particles' positions and velocities
    particles = np.random.randint(1, batch_size, size=(num_particles, 2))
    velocities = np.random.uniform(-1, 1, size=(num_particles, 2))
    
    # Initialize personal best positions and scores
    pbest_positions = np.copy(particles)
    pbest_scores = np.full(num_particles, -np.inf)
    
    # Initialize global best position and score
    gbest_position = None
    gbest_score = -np.inf
    
    for i in range(max_iter):
        for j in range(num_particles):
            # Evaluate the fitness of the current particle
            x, y = particles[j]
            score = evaluation_fn(traces, x, y, batch_size=batch_size)
            
            # Update personal best if the current score is better
            if score > pbest_scores[j]:
                pbest_positions[j] = particles[j]
                pbest_scores[j] = score
            
            # Update global best if the current score is better
            if score > gbest_score:
                gbest_position = particles[j]
                gbest_score = score
        
        # Update velocities and positions of particles
        for j in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[j] = (
                w * velocities[j] +
                c1 * r1 * (pbest_positions[j] - particles[j]) +
                c2 * r2 * (gbest_position - particles[j])
            )
            particles[j] = np.clip(particles[j] + velocities[j].astype(int), 1, batch_size-1)
    
    best_x, best_y = gbest_position
    return best_x, best_y, gbest_score