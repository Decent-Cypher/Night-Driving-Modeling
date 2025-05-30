from numpy._typing._array_like import NDArray
from numpy import float64
import numpy as np
from typing import Callable, Literal, final, overload
from tqdm import tqdm
import matplotlib.pyplot as plt

SYSTEM_LENGTH = 500.0

def normal_optimal_velocity(delta_x:NDArray[np.float64], x_c:float=2.0) -> NDArray[float64]:
    """
    Calculate the optimal velocity based on the distance to the next car.
    
    Parameters:
    - delta_x: array of headway distances
    - x_c: constant used in tanh-based velocity model
    
    Returns:
    - Array of optimal velocities
    """
    return np.asarray(np.tanh(delta_x - x_c) + np.tanh(x_c), dtype=np.float64)

def night_optimal_velocity(delta_x:NDArray[np.float64], x_c:float=2.0, a:float=5.0, b:float=1.0, x_c1:float=3.2, x_c2:float=4.0) -> NDArray[float64]:
    """
    Calculate the optimal velocity based on the distance to the next car during night time.
    
    Parameters:
    - delta_x: array of headway distances
    - x_c: constant used in tanh-based velocity model
    - a, b: constants defining linear and fixed velocities
    - x_c1, x_c2: thresholds defining the piecewise function
    
    Returns:
    - Array of optimal velocities
    """
    V = np.zeros_like(delta_x)
    # Region 1: delta_x < x_c1
    mask1 = delta_x < x_c1
    V[mask1] = np.tanh(delta_x[mask1] - x_c) + np.tanh(x_c)
    # Region 2: x_c1 <= delta_x <= x_c2
    mask2 = (delta_x >= x_c1) & (delta_x <= x_c2)
    V[mask2] = a - delta_x[mask2]
    # Region 3: delta_x > x_c2
    mask3 = delta_x > x_c2
    V[mask3] = b

    return V
    
def FVD_acceleration(positions: NDArray[np.float64], velocities: NDArray[np.float64], optimal_velocity_function:Callable[[NDArray[np.float64]], NDArray[np.float64]], kappa:float, _lambda:float) -> NDArray[np.float64]:
    """
    Calculate the acceleration of each car based on their positions and velocities.
    
    Parameters:
    - positions: array of car positions
    - velocities: array of car velocities
    - optimal_velocity_function: function to calculate optimal velocity
    - kappa: sensitivity parameter for the optimal velocity
    - _lambda: sensitivity parameter for the relative velocity
    
    Returns:
    - Array of accelerations for each car
    """
    # Compute headways with periodic boundary conditions
    delta_x = np.roll(positions, -1) - positions
    delta_x[-1] = SYSTEM_LENGTH - positions[-1] + positions[0]  # Wrap around for the last car
    print(delta_x)
    # Compute optimal velocities for each headway
    V = optimal_velocity_function(delta_x)
    print(V)
    # Compute velocity differences between each vehicle and its leader
    delta_v = np.roll(velocities, -1) - velocities
    print(delta_v)
    # Compute accelerations
    accelerations = kappa * (V - velocities) + _lambda * delta_v
    print(accelerations)

    return accelerations

@final
class FVD_Simulator:
    def __init__(self, positions: NDArray[np.float64], velocities: NDArray[np.float64], optimal_velocity_function:Callable[[NDArray[np.float64]], NDArray[np.float64]], kappa: float, _lambda: float, delta_t: float=0.1):
        """
        Initialize the simulator with car positions, velocities, and an optimal velocity function.
        
        Parameters:
        - positions: initial positions of the cars
        - velocities: initial velocities of the cars
        - optimal_velocity_function: function to calculate optimal velocity
        """
        self.positions = positions
        self.velocities = velocities
        self.optimal_velocity_function = optimal_velocity_function
        self.kappa = kappa
        self._lambda = _lambda
        self.delta_t = delta_t
    def step(self):
        """
        Perform a single simulation step, updating positions and velocities.
        """
        # Calculate accelerations
        accelerations = FVD_acceleration(self.positions, self.velocities, self.optimal_velocity_function, self.kappa, self._lambda)
        # Update velocities
        self.velocities += accelerations * self.delta_t
        # Update positions
        self.positions += self.velocities * self.delta_t + 0.5 * accelerations * self.delta_t**2
        self.positions[self.positions >= SYSTEM_LENGTH] -= SYSTEM_LENGTH  # Wrap around positions
        
    @overload
    def run(self, steps: int, log_result:Literal[False]) -> None: ...
    
    @overload
    def run(self, steps: int, log_result:Literal[True]) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]: ...
    
    def run(self, steps: int, log_result:bool=False) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]] | None:
        """
        Run the simulation for a specified number of steps.
        
        Parameters:
        - steps: number of simulation steps to run
        - log_result: whether to log the positions and velocities at each step
        
        Returns:
        - If log_result is True, returns a tuple of lists containing positions and velocities at each step.
        - If log_result is False, returns None.
        """
        if log_result:
            log_positions = [self.positions.copy()]
            log_velocities = [self.velocities.copy()]
            for _ in tqdm(range(steps), desc="Running FVD Simulation"):
                self.step()
                log_positions.append(self.positions.copy())
                log_velocities.append(self.velocities.copy())
            
            return log_positions, log_velocities
        else:
            for _ in tqdm(range(steps), desc="Running FVD Simulation"):
                self.step()
            return None

if __name__ == '__main__':
    from functools import partial
    test_optimal_velocity = partial(night_optimal_velocity, x_c=2.0, a=5.0, b=1.0, x_c1=3.2, x_c2=4.0)
    test_positions = np.array([450. , 451.2, 455. , 460. ]).astype(np.float64)
    print("Test positions:", test_positions)
    test_velocities = np.array([1, 1.8, 2.2, 5]).astype(np.float64)
    print("Test velocities:", test_velocities)
    sim = FVD_Simulator(
        positions=test_positions,
        velocities=test_velocities,
        optimal_velocity_function=test_optimal_velocity,
        kappa=1,
        _lambda=0.5,
        delta_t=0.1
    )
    log_p, log_v = sim.run(steps=500, log_result=True)
        
    # Number of positions per array
    num_positions = 4

    # Generate a colormap
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / num_positions) for i in range(num_positions)]

    # Iterate over each position index
    for i in range(num_positions):
        x = [idx for idx, _ in enumerate(log_p)]
        y = [positions[i] for positions in log_p]
        plt.scatter(x, y, color=colors[i], label=f'Position {i+1}')

    # Customize the plot
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Car Positions Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('car_positions_over_time.png')
