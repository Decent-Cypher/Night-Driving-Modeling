from fvd_simulator import (
    night_optimal_velocity,
    normal_optimal_velocity, # pyright: ignore[reportUnusedImport]
    plot_fundamental_diagram,
)
from functools import partial
# import numpy as np


if __name__ == "__main__":
    test_night_optimal_velocity = partial(night_optimal_velocity, x_c=2.0, a=5.0, b=1.0, x_c1=3.2, x_c2=4.0)
    SIMULATION_STEPS = 500
    # plot_fundamental_diagram(
    #     optimal_velocity_function=normal_optimal_velocity,
    #     simulation_steps=SIMULATION_STEPS,
    #     kappa=1,
    #     _lambda=0.1,
    #     delta_t=0.1,
    #     filename="fundamental_diagram_normal_fig6.png",
    # )
    # plot_fundamental_diagram(
    #     optimal_velocity_function=test_night_optimal_velocity,
    #     simulation_steps=SIMULATION_STEPS,
    #     kappa=1,
    #     _lambda=0.1,
    #     delta_t=0.1,
    #     filename="fundamental_diagram_night_fig6.png",
    # )
    plot_fundamental_diagram(
        optimal_velocity_function=test_night_optimal_velocity,
        simulation_steps=SIMULATION_STEPS,
        kappa=1,
        _lambda=0.1,
        n_dec=1,
        delta_t=0.4,
        filename="fundamental_diagram_night_fig6_perturb.png",
    )