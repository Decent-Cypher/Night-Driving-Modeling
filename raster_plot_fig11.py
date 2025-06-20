from fvd_simulator import (
    FVD_Simulator_with_Perturbation,
    night_optimal_velocity,
    SYSTEM_LENGTH,
    plot_traffic_raster
)
from functools import partial
import numpy as np


if __name__ == "__main__":
    test_optimal_velocity = partial(night_optimal_velocity, x_c=2.0, a=5.0, b=1.0, x_c1=3.2, x_c2=4.0)
    N = 220
    SIMULATION_STEPS = 1000
    test_positions = np.random.uniform(low=0.0, high=SYSTEM_LENGTH, size=N).astype(np.float64)
    test_positions.sort()
    test_velocities = np.ones_like(test_positions, dtype=np.float64) * 0.1
    
    sim = FVD_Simulator_with_Perturbation(
        positions=test_positions,
        velocities=test_velocities,
        optimal_velocity_function=test_optimal_velocity,
        kappa=1,
        _lambda=0.5,
        n_dec=80,
        delta_t=0.1,
    )
    log_p, log_v = sim.run(steps=SIMULATION_STEPS, log_result=True)
    
    plot_traffic_raster(log_p, figsize=(6, 8), filename='raster_plot_fig11.png')
    