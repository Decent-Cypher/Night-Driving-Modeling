from fvd_simulator import (
    FVD_Simulator,
    night_optimal_velocity,
    SYSTEM_LENGTH,
    plot_traffic_raster
)
from functools import partial
import numpy as np


if __name__ == "__main__":
    test_optimal_velocity = partial(night_optimal_velocity, x_c=2.0, a=5.0, b=1.0, x_c1=3.2, x_c2=4.0)
    N = 150
    SIMULATION_STEPS = 1000
    test_positions = np.random.uniform(low=0.0, high=SYSTEM_LENGTH, size=N).astype(np.float64)
    test_positions.sort()
    test_velocities = np.ones_like(test_positions, dtype=np.float64) * 0.2
    
    sim = FVD_Simulator(
        positions=test_positions,
        velocities=test_velocities,
        optimal_velocity_function=test_optimal_velocity,
        kappa=1,
        _lambda=0.5,
        delta_t=0.1,
    )
    log_p, log_v = sim.run(steps=SIMULATION_STEPS, log_result=True)
    
    plot_traffic_raster(log_p, figsize=(6, 8), filename='output_images/raster_plot_fig3.png')
    
    # Plot the average velocity over time
    import matplotlib.pyplot as plt
    avg_velocity = [np.mean(log_v[i]).item() for i in range(len(log_v))]
    plt.figure(figsize=(6, 4))
    plt.plot(avg_velocity, label='Average Velocity', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Average Velocity')
    plt.title('Average Velocity Over Time')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('output_images/average_velocity_over_time_fig3.png', dpi=300, bbox_inches='tight')