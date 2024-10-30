import numpy as np

def do_swap(sol: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Swap two tasks in the solution schedule to generate a neighbor solution.
    """
    if i != j:  # Ensure that indices are different
        sol[i], sol[j] = sol[j], sol[i]
    return sol


def local_search(problem, init_sol: np.ndarray, verbose: bool = False) -> (np.ndarray, float):
    """
    Perform a local search to minimize the makespan of the given problem starting from an initial solution.

    Args:
        problem: An object with `get_dim()` and `calculate_makespan()` methods.
        init_sol (np.ndarray): The initial solution array.
        verbose (bool): If True, prints detailed information about the search process.

    Returns:
        (np.ndarray, float): The best solution found and its makespan.
    """
    n = problem.get_dim()
    best_sol = init_sol.copy()
    best_makespan, _ = problem.calculate_makespan(best_sol)

    if verbose:
        print(f"Initial makespan: {best_makespan}")

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Swap and evaluate the new solution
                candidate_sol = do_swap(best_sol, i, j)
                candidate_makespan, _ = problem.calculate_makespan(candidate_sol)

                # If the new solution is better, update the best solution
                if candidate_makespan < best_makespan:
                    best_sol = candidate_sol
                    best_makespan = candidate_makespan
                    improved = True
                    if verbose:
                        print(f"New best makespan: {best_makespan} by swapping indices {i} and {j}")
                    break  # Restart the search after improvement
            if improved:
                break  # Restart the search after improvement

    return best_sol, best_makespan