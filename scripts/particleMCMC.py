import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, uniform
from tqdm import tqdm

from dataGen import generate_data
from particleFilter import motion_model, measurement_model, resample, likelihoods_est, particle_filter

def particle_marginal_metropolis_hastings(observations, 
                                          init_A, init_Q, 
                                          init_c1, 
                                          init_c2, 
                                          init_r, 
                                          initial_particles, 
                                          num_iterations, 
                                          stepsize=0.1):
    T = len(observations)
    num_particles = len(initial_particles)
    particles = initial_particles.copy()
    logLikelihoods = np.zeros(num_iterations)
    
    A = np.zeros(num_iterations)
    A_proposed = np.zeros(num_iterations)
    
    Q = np.zeros(num_iterations)
    Q_proposed = np.zeros(num_iterations)
    
    r = np.zeros(num_iterations)
    r_proposed = np.zeros(num_iterations)
    
    c1 = np.zeros(num_iterations)
    c2 = np.zeros(num_iterations)
    c1_proposed = np.zeros(num_iterations)
    c2_proposed = np.zeros(num_iterations)

    # Step 1
    A[0] = init_A
    Q[0] = init_Q
    r[0] = init_r
    R_ = np.array([[init_r, 0], [0, init_r]])
    C_ = np.array([[init_c1, init_c2]])
    estimated_state, logLikelihoods[0], _ = particle_filter(observations, A[0], Q[0], C_, R_, initial_particles)
    
    for iteration in range(1, num_iterations):
        # Sample a new parameter
        A_proposed[iteration] = max(0.0, np.random.normal(A[iteration - 1], stepsize, 1))
        Q_proposed[iteration] = max(0.0, np.random.normal(Q[iteration - 1], stepsize, 1))
        r_proposed[iteration] = max(0.0, np.random.normal(r[iteration - 1], stepsize, 1))
        c1_proposed[iteration] = max(0.0, np.random.normal(c1[iteration - 1], stepsize, 1))
        c2_proposed[iteration] = max(0.0, np.random.normal(c2[iteration - 1], stepsize, 1))
        
        R_ = np.array([[r_proposed[iteration], 0], [0, r_proposed[iteration]]])
        C_ = np.array([[c1_proposed[iteration], c2_proposed[iteration]]])

        # Apply particle filter
        estimated_state, logLikelihoods[iteration], weights = particle_filter(observations, 
                                                                              A_proposed[iteration], 
                                                                              Q_proposed[iteration], 
                                                                              C_,
                                                                              R_, 
                                                                              initial_particles)
        # Calculate the acceptance probability
        log_acceptance_prob = (logLikelihoods[iteration] + 
                               uniform.logpdf(A_proposed[iteration], 0, 2) + 
                               uniform.logpdf(Q_proposed[iteration], 0, 5) + 
                               uniform.logpdf(r_proposed[iteration], 0, 2) + 
                               uniform.logpdf(c1_proposed[iteration], 0, 2) + 
                               uniform.logpdf(c2_proposed[iteration], 0, 2)) -\
                              (logLikelihoods[iteration - 1] + 
                               uniform.logpdf(A[iteration - 1], 0, 2) + 
                               uniform.logpdf(Q[iteration - 1], 0, 5) +
                               uniform.logpdf(r[iteration - 1], 0, 2) + 
                               uniform.logpdf(c1[iteration], 0, 2) + 
                               uniform.logpdf(c2[iteration], 0, 2))

        acceptance_prob = np.min((1.0, np.exp(log_acceptance_prob)))

        if np.random.uniform() < acceptance_prob:
            A[iteration] = A_proposed[iteration]
            Q[iteration] = Q_proposed[iteration]
            r[iteration] = r_proposed[iteration]
            c1[iteration] = c1_proposed[iteration]
            c2[iteration] = c2_proposed[iteration]
        else:
            A[iteration] = A[iteration - 1]
            Q[iteration] = Q[iteration - 1]
            r[iteration] = r[iteration - 1]
            c1[iteration] = c1[iteration - 1]
            c2[iteration] = c2[iteration - 1]

    
    return A, Q, r, c1, c2

if __name__ == "__main__":
    
    param_state_est = True

    # Define parameters
    T = 100  # Number of time steps
    A = .9  # Transition matrix
    Q = 3.2  # Covariance matrix of the state noise
    C = np.array([[1.5, 0.6]])  # Observation matrix
    R = np.array([[1, 0], [0, 1]])  # Covariance matrix of the observation noise
    initial_state = np.array([np.random.uniform(-1,-1)])#np.array([0])  # Initial state
    initial_covariance = np.array([[1]])  # Initial state covariance
    initial_particles = np.random.normal(0, 1, size=50)  # Initial particles
    num_iterations = 100000  # Number of PMMH iterations

    # Generate synthetic data
    latent_states, observations = generate_data(T, A, Q, C, R, initial_state)
    
    if not param_state_est:
        # Apply particle filter
        estimated_state, loglik, _ = particle_filter(observations, A, Q, C, R, initial_particles)
        
        plt.plot(latent_states)
        plt.plot(estimated_state)
        #plt.show()
    else:
        # Apply particle marginal Metropolis-Hastings algorithm
        A_init = .1
        Q_init = .1
        r_init = .1
        c1_init = .1
        c2_init = .1
        estimated_A, estimated_Q, estimated_r, estimated_c1, estimated_c2 = particle_marginal_metropolis_hastings(
            observations, 
            A_init, 
            Q_init, 
            c1_init, 
            c2_init, 
            r_init, 
            initial_particles, 
            num_iterations
        )
    
    # Create a figure and two subplots side by side
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)

    ax1.plot(estimated_A)
    ax1.set_title('A_est')

    ax2.plot(estimated_Q)
    ax2.set_title('Q_est')

    ax3.plot(estimated_r)
    ax3.set_title('r_est')

    ax4.plot(estimated_c1)
    ax4.set_title('c1_est')

    ax5.plot(estimated_c2)
    ax5.set_title('c2_est')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.5)
    #plt.plot(estimated_Q)