import numpy as np
import pandas as pd
import pprint

from tqdm import tqdm

def generate_data(T, A, Q, C, R, initial_state):
    latent_states = [initial_state]
    observations = [np.multiply(C, initial_state) + np.random.multivariate_normal(np.zeros(2), R)]
    
    for _ in range(T-1):
        latent_state = np.dot(A, latent_states[-1]) + np.random.normal(np.zeros(1), Q)
        observation = np.multiply(C, latent_state) + np.random.multivariate_normal(np.zeros(2), R)
        
        latent_states.append(latent_state)
        observations.append(observation)
    
    return latent_states, observations

if __name__ == "__main__":
    # Define parameters
    T = 10  # Number of time steps
    A = 0.8  # Transition matrix
    Q = 0.1  # Covariance matrix of the state noise
    C = np.array([[1, 0]])  # Observation matrix
    R = np.array([[0.5, 0], [0, 0.5]])  # Covariance matrix of the observation noise
    initial_state = np.array([0])  # Initial state
    initial_covariance = np.array([[1]])  # Initial state covariance

    # Generate synthetic data
    latent_states, observations = generate_data(T, A, Q, C, R, initial_state)

    # Apply Kalman filter
    #filtered_states, filtered_covariances = kalman_filter(observations, A, Q, C, R, initial_state, initial_covariance)

    # Print results
    print("True latent states:", latent_states)
    print("Observations:", observations)
    #print("Filtered latent states:", filtered_states)

    '''
    states = {}
    observations = {}

    experiments = []
    times = []
    obs1 = []
    obs2 = []

    for i in tqdm(range(n_experiments)):
        initState = np.array([0.0])
        x, y = generateData(theta_x, theta_y, n_obs, initState)
        states[i] = x
        experiments += [i for _ in range(n_obs)]
        times += [i for i in range(n_obs)]
        obs1 += y[0, :].tolist()
        obs2 += y[1, :].tolist()

    observations["experiment_" + str(i)] = experiments
    observations["time"] = times
    observations["obs1"] = obs1
    observations["obs2"] = obs2
    print(pd.DataFrame(observations))
    '''