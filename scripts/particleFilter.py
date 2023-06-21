import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from tqdm import tqdm

from dataGen import generate_data

# Function for particle filtering with linear dynamical system
def motion_model(particles, A, Q):
    num_particles = len(particles)
    noise = np.random.normal(np.zeros(1), Q, size=num_particles)
    particles = np.dot(A, particles.T).T + noise
    return particles

def measurement_model(particles, C, R):
    num_particles = len(particles)
    noise = np.random.multivariate_normal(np.zeros(2), R, size=num_particles)
    measurements = np.dot(C.T, np.reshape(particles, (1, num_particles))).T + noise
    return measurements

def likelihoods_est(y_hat, y):
    return -0.5 * np.sum((y_hat - y)**2, axis=1)

def resample(particles, weights):
    num_particles = len(particles)
    indices = np.random.choice(range(num_particles), size=num_particles, replace=True, p=weights)
    resampled_particles = particles[indices]
    resampled_weights = np.ones(num_particles) / num_particles
    return resampled_particles, resampled_weights

def particle_filter(observations, A, Q, C, R, initial_particles):
    T = len(observations)
    num_particles = len(initial_particles)
    particles = initial_particles.copy()
    weights_ = np.zeros((num_particles, T))
    weights = np.ones(num_particles) / num_particles
    weights_[:,0] = weights
    x_hat = []
   
    for t in range(1,T):
        # Motion update
        particles = motion_model(particles, A, Q)
        
        # Measurement update
        measurements_predicted = measurement_model(particles, C, R)
        likelihoods = likelihoods_est(measurements_predicted, observations[t])
        weights_[:,t] = np.log(weights_[:,t-1]) + likelihoods
        max_weight = np.max(weights_[:,t])
        weights_[:,t] = np.exp(weights_[:,t] - max_weight)
        weights_[:,t] = weights_[:,t]/np.sum(weights_[:,t])
        
        # Resampling
        particles, weights_[:,t] = resample(particles, weights_[:,t])
        
        # Estimate the current state
        #estimated_state = np.sum(particles * weights, axis=None)
        estimated_state = np.sum(particles * weights_[:,t], axis=None)
        x_hat.append(estimated_state)
        
    return x_hat, likelihoods, weights_


if __name__ == "__main__":
    print("This is a module for particle filtering.")

    # Define parameters
    T = 100  # Number of time steps
    A = .9  # Transition matrix
    Q = 1  # Covariance matrix of the state noise
    C = np.array([[1.02, 0]])  # Observation matrix
    R = np.array([[1, 0], [0, 1]])  # Covariance matrix of the observation noise
    initial_state = np.array([np.random.uniform(-1,-1)])#np.array([0])  # Initial state
    initial_covariance = np.array([[1]])  # Initial state covariance
    initial_particles = np.random.normal(0, 1, size=100)  # Initial particles

    # Generate synthetic data
    latent_states, observations = generate_data(T, A, Q, C, R, initial_state)

    # Apply particle filter
    estimated_state = particle_filter(observations, A, Q, C, R, initial_particles)

    plt.plot(latent_states)
    plt.plot(estimated_state)
    #plt.show()