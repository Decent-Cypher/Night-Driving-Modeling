from numpy._typing._array_like import NDArray
import numpy as np
from typing import Callable, Literal, final, overload
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SYSTEM_LENGTH = 500.0

def calculate_flow(positions_list: list[NDArray[np.float64]], velocities_list: list[NDArray[np.float64]]) -> float:
    """
    Calculate the flow of cars based on their positions and velocities.
    
    Parameters:
    - positions: array of car positions
    - velocities: array of car velocities
    
    Returns:
    - Flow rate (cars per second)
    """
    velocities = np.array(velocities_list, dtype=np.float64)
    positions = np.array(positions_list, dtype=np.float64)
    
    # Ensure positions are within the system length
    assert np.all(positions >= 0) and np.all(positions < SYSTEM_LENGTH), "Positions must be within the system length."

    # Calculate the average velocity
    avg_velocity = np.mean(velocities[100:]).item()
    
    # Calculate the density (cars per unit length)
    density = positions.shape[1] / SYSTEM_LENGTH
    
    # Flow is density times average velocity
    flow = density * avg_velocity
    
    # print(f"Average velocity: {avg_velocity}, Density: {density}, Flow: {flow}")
    return flow

def plot_traffic_raster(positions_list: list[NDArray[np.float64]], figsize: tuple[int, int] = (6, 8), filename:str='traffic_raster_plot.png') -> None:
    """
    Generate a binary traffic raster plot with correct axis mapping.

    Parameters:
    - positions_list: list of 1D arrays (or similarly shaped 2D),
      shaped (T, L), where each entry is 1 if a car is present, else 0.
    - system_length: total road length (0 to system_length) for x-axis.
    - cmap: colormap ('Greys' gives 0=black, 1=white).
    - figsize: output figure size.
    """
    positions_list = [positions_list[i] for i in range(0, len(positions_list), 2)]
    T = len(positions_list)
    L = int(SYSTEM_LENGTH)
    mat = np.zeros((T, L), dtype=np.uint8)  # create white background

    for t, pos in enumerate(positions_list):
        idx = np.rint(pos).astype(int)
        idx = np.clip(idx, 0, L - 1)  # keep positions in range
        mat[t, idx] = 1  # mark black for each car

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(mat, cmap='gray_r', origin='lower',
              interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Position (0 to SYSTEM_LENGTH)')
    ax.set_ylabel('Time step')
    ax.set_title('Car positions over time')
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, T - 0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def animate_simulation(positions_list:list[NDArray[np.float64]], interval:int=200):
    """
    Animate car positions over time.
    
    positions: ndarray of shape (T, N) — car positions at each time step
    colors: list or array of length N, specifying color for each car (optional)
    interval: delay between frames in ms
    """
    T = len(positions_list)
    N = positions_list[0].shape[0]
    y = np.zeros(N)  # y-axis all zeros (positions on a line)
    
    # Assign colors per car
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(N)]
    
    fig, ax = plt.subplots()
    # Initialize scatter with first frame
    scat = ax.scatter(positions_list[0], y, c=colors, s=100)

    # Setting consistent axis range
    all_positions = np.concatenate(positions_list)
    ax.set_xlim(all_positions.min() - 1, all_positions.max() + 1) # pyright: ignore[reportAny]
    ax.set_ylim(-1, 1)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Position along line")
    ax.set_title("Time step: 0")

    def update(frame): # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        data = np.c_[positions_list[frame], y] # pyright: ignore[reportAny]
        scat.set_offsets(data)# pyright: ignore[reportAny]
        ax.set_title(f"Time step: {frame}")
        return scat,

    ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
    return ani

def normal_optimal_velocity(delta_x:NDArray[np.float64], x_c:float=2.0) -> NDArray[np.float64]:
    """
    Calculate the optimal velocity based on the distance to the next car.
    
    Parameters:
    - delta_x: array of headway distances
    - x_c: constant used in tanh-based velocity model
    
    Returns:
    - Array of optimal velocities
    """
    return np.asarray(np.tanh(delta_x - x_c) + np.tanh(x_c), dtype=np.float64)

def night_optimal_velocity(delta_x:NDArray[np.float64], x_c:float=2.0, a:float=5.0, b:float=1.0, x_c1:float=3.2, x_c2:float=4.0) -> NDArray[np.float64]:
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
    # Assert that only one values is negative
    assert np.sum(delta_x < 0) == 1, "There should be exactly one negative headway."
    delta_x[delta_x < 0] += SYSTEM_LENGTH  # Adjust negative headway to wrap around
    # delta_x[-1] = SYSTEM_LENGTH - positions[-1] + positions[0]  # Wrap around for the last car
    # Compute optimal velocities for each headway
    V = optimal_velocity_function(delta_x)
    # Compute velocity differences between each vehicle and its leader
    delta_v = np.roll(velocities, -1) - velocities
    # Compute accelerations
    accelerations = kappa * (V - velocities) + _lambda * delta_v

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
        # Velocities must be non-negative
        assert np.all(velocities >= 0), "Velocities must be non-negative."
        # Positions must be within the system length
        assert np.all(positions >= 0) and np.all(positions < SYSTEM_LENGTH), "Positions must be within the system length."
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
        # Update positions
        self.positions = np.maximum(self.positions, self.positions + self.velocities * self.delta_t + 0.5 * accelerations * self.delta_t**2)
        # Update velocities
        self.velocities += accelerations * self.delta_t
        # Ensure positions are circular around the system length
        self.velocities = np.clip(self.velocities, 0, None)  # Ensure non-negative velocities
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

@final
class FVD_Simulator_with_Randomness:
    def __init__(self, positions: NDArray[np.float64], velocities: NDArray[np.float64], optimal_velocity_function:Callable[[NDArray[np.float64]], NDArray[np.float64]], kappa: float, _lambda: float, delta_t: float=0.1, A: float=0.1):
        """
        Initialize the simulator with car positions, velocities, and an optimal velocity function.
        
        Parameters:
        - positions: initial positions of the cars
        - velocities: initial velocities of the cars
        - optimal_velocity_function: function to calculate optimal velocity
        """
        # Velocities must be non-negative
        assert np.all(velocities >= 0), "Velocities must be non-negative."
        # Positions must be within the system length
        assert np.all(positions >= 0) and np.all(positions < SYSTEM_LENGTH), "Positions must be within the system length."
        self.positions = positions
        self.velocities = velocities
        self.optimal_velocity_function = optimal_velocity_function
        self.kappa = kappa
        self._lambda = _lambda
        self.delta_t = delta_t
        self.A = A # Amplitude of the randomness
    def step(self):
        """
        Perform a single simulation step, updating positions and velocities.
        """
        # Calculate accelerations
        accelerations = FVD_acceleration(self.positions, self.velocities, self.optimal_velocity_function, self.kappa, self._lambda)
        # Update velocities and positions
        old_velocities = self.velocities.copy()
        self.velocities += accelerations * self.delta_t + np.random.uniform(-0.5, 0.5, size=self.velocities.shape) * self.A  # Add randomness to velocities
        self.velocities = np.clip(np.clip(self.velocities, 0, None), None, self.optimal_velocity_function(np.array(3.2)).item())
        thres = np.roll(self.positions, -1)
        thres[thres < np.roll(thres, 1)] += SYSTEM_LENGTH  # Adjust positions for periodic boundary conditions
        self.positions = np.minimum(np.maximum(self.positions, self.positions + 0.5 * (old_velocities + self.velocities) * self.delta_t), thres)  # Update positions using average velocity
        
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

@final
class FVD_Simulator_with_Perturbation:
    def __init__(self, positions: NDArray[np.float64], velocities: NDArray[np.float64], optimal_velocity_function:Callable[[NDArray[np.float64]], NDArray[np.float64]], kappa: float, _lambda: float, delta_t: float=0.1, n_dec:int=1):
        """
        Initialize the simulator with car positions, velocities, and an optimal velocity function.
        
        Parameters:
        - positions: initial positions of the cars
        - velocities: initial velocities of the cars
        - optimal_velocity_function: function to calculate optimal velocity
        """
        # Velocities must be non-negative
        assert np.all(velocities >= 0), "Velocities must be non-negative."
        # Positions must be within the system length
        assert np.all(positions >= 0) and np.all(positions < SYSTEM_LENGTH), "Positions must be within the system length."
        self.positions = positions
        self.velocities = velocities
        self.optimal_velocity_function = optimal_velocity_function
        self.kappa = kappa
        self._lambda = _lambda
        self.delta_t = delta_t
        assert n_dec >= 1, "n_dec must be at least 1."
        self.n_dec = n_dec
        self.start_perturb = False
    def step(self):
        """
        Perform a single simulation step, updating positions and velocities.
        """
        # Calculate accelerations
        accelerations = FVD_acceleration(self.positions, self.velocities, self.optimal_velocity_function, self.kappa, self._lambda)
        # Perturbation will change the acceleration of the car at the perturb_index to -1
        if self.n_dec > 0:
            perturb_index = np.random.choice(len(self.positions), size=1, replace=False).item()
            accelerations[perturb_index] = -1
        
        # Update positions
        self.positions = np.maximum(self.positions, self.positions + self.velocities * self.delta_t + 0.5 * accelerations * self.delta_t**2)
        # Update velocities
        self.velocities += accelerations * self.delta_t
        # Ensure positions are circular around the system length
        self.velocities = np.clip(self.velocities, 0, None)  # Ensure non-negative velocities
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
            for i in tqdm(range(steps), desc="Running FVD Simulation"):
                self.step()
                log_positions.append(self.positions.copy())
                log_velocities.append(self.velocities.copy())
                if i >= 100:
                    self.start_perturb = True
            
            return log_positions, log_velocities
        else:
            for i in tqdm(range(steps), desc="Running FVD Simulation"):
                self.step()
                if i >= 100:
                    self.start_perturb = True
                
            return None

def plot_fundamental_diagram(optimal_velocity_function:Callable[[NDArray[np.float64]], NDArray[np.float64]], simulation_steps:int=1000, kappa:float=1.0, _lambda:float=0.5, delta_t:float=0.1, n_dec:int=0, debug_checkpoint:int=-1, filename:str='fundamental_diagram.png'):
    """
    Run a test simulation with the given parameters.
    
    Parameters:
    - optimal_velocity_function: function to calculate optimal velocity
    - kappa: sensitivity parameter for the optimal velocity
    - _lambda: sensitivity parameter for the relative velocity
    - delta_t: time step for the simulation
    - n_dec: number of perturbations to apply
    """
    y_axis = []
    x_axis = list(range(2, 500))
    for N in tqdm(x_axis, desc="Running Test Simulations"):
        # positions = np.random.uniform(low=0.0, high=SYSTEM_LENGTH, size=N).astype(np.float64)
        # positions.sort()
        positions = np.linspace(0, SYSTEM_LENGTH-SYSTEM_LENGTH/N, N).astype(np.float64)  # Evenly spaced positions
        velocities = np.ones_like(positions, dtype=np.float64) * 0.05
        
        sim = None
        if n_dec == 0:
            sim = FVD_Simulator(
                positions=positions,
                velocities=velocities,
                optimal_velocity_function=optimal_velocity_function,
                kappa=kappa,
                _lambda=_lambda,
                delta_t=delta_t,
            )
        else:
            # Use the perturbation simulator
            sim = FVD_Simulator_with_Perturbation(
                positions=positions,
                velocities=velocities,
                optimal_velocity_function=optimal_velocity_function,
                kappa=kappa,
                _lambda=_lambda,
                delta_t=delta_t,
                n_dec=n_dec,
            )
        
        log_p, log_v = sim.run(steps=simulation_steps, log_result=True)
        
        flow = calculate_flow(log_p, log_v)
        y_axis.append(flow)
        if debug_checkpoint > 0 and N == debug_checkpoint:
            plot_traffic_raster(log_p, figsize=(6, 8), filename=filename[:-4] + f'_debug_{debug_checkpoint}.png')
    # Convert the number of cars to density
    x_axis = [item / SYSTEM_LENGTH for item in x_axis]
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, y_axis, marker='o', linestyle='-', color='b')
    plt.title('Fundamental Diagram')
    plt.xlabel('Density (cars per unit length)')
    plt.ylabel('Flow (cars per timestep)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

if __name__ == '__main__':
    from functools import partial
    test_optimal_velocity = partial(night_optimal_velocity, x_c=2.0, a=5.0, b=1.0, x_c1=3.2, x_c2=4.0)
    # test_optimal_velocity = normal_optimal_velocity  # Use the normal optimal velocity function for testing
    # test_positions = np.array([450. , 451.2, 455. , 460. ]).astype(np.float64)
    # SIMULATION_STEPS = 500
    # print("Test positions:", test_positions)
    # test_velocities = np.array([1, 1.8, 2.2, 5]).astype(np.float64)
    # print("Test velocities:", test_velocities)
    
    N = 150
    SIMULATION_STEPS = 500
    plot_fundamental_diagram(
        optimal_velocity_function=test_optimal_velocity,
        simulation_steps=SIMULATION_STEPS,
        kappa=1,
        _lambda=0.2,
        delta_t=0.3,
        n_dec=1,  # Set to 0 for normal simulation, >0 for perturbation simulation,
        debug_checkpoint=250
    )
    
    
    
