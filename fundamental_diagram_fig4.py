from fvd_simulator import (
    night_optimal_velocity,
    normal_optimal_velocity,
    plot_fundamental_diagram,
)
from functools import partial
# import numpy as np


if __name__ == "__main__":
    test_night_optimal_velocity = partial(night_optimal_velocity, x_c=2.0, a=5.0, b=1.0, x_c1=3.2, x_c2=4.0)
    SIMULATION_STEPS = 500
    plot_fundamental_diagram(
        optimal_velocity_function=normal_optimal_velocity,
        simulation_steps=SIMULATION_STEPS,
        kappa=1,
        _lambda=0.2,
        delta_t=0.1,
        filename="output_images/fundamental_diagram_normal_fig4.png",
    )
    plot_fundamental_diagram(
        optimal_velocity_function=test_night_optimal_velocity,
        simulation_steps=SIMULATION_STEPS,
        kappa=1,
        _lambda=0.2,
        delta_t=0.1,
        filename="output_images/fundamental_diagram_night_fig4.png",
    )
    plot_fundamental_diagram(
        optimal_velocity_function=test_night_optimal_velocity,
        simulation_steps=SIMULATION_STEPS,
        kappa=1,
        _lambda=0.2,
        n_dec=1,
        delta_t=0.35,
        filename="output_images/fundamental_diagram_night_fig4_perturb.png",
    )