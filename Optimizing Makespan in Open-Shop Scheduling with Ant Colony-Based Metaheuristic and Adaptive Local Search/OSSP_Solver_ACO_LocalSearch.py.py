import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import json
from statistics import mean
from typing import List, Tuple, Dict, Optional
import logging
import time
from local_search.iterated_local_search import *
from alive_progress import alive_bar
from pathlib import Path


class OSSP_problem:
    def __init__(self, instance):
        """
        Initialize the OSSP problem with the given instance data.
        
        Args:
            instance: A 2D array representing the execution times for each job on each machine.
        """
        self.data = instance
        self.number_of_jobs = len(self.data)
        self.number_of_machines = len(self.data[0])

    @staticmethod
    def load_instance(file_name, verbose=True):
        """
        Load an OSSP problem instance from a file.
        
        Args:
            file_name (str): Name of the file containing the instance data.
            verbose (bool): If True, print information about the loaded instance.
        
        Returns:
            OSSP_problem: An instance of OSSP_problem with the loaded data.
        """
        file_path = script_dir / "test_instances" / file_name   
        with file_path.open("r") as f:
            lines = f.readlines()
        
        number_of_jobs = len(lines)
        number_of_machines = len(lines[0].split())
        
        instance = np.zeros((number_of_jobs, number_of_machines), dtype=int)
        
        for i, line in enumerate(lines):
            values = list(map(int, line.split()))
            instance[i, :] = values

        if verbose:
            print(f"Number of jobs: {number_of_jobs}")
            print(f"Number of machines: {number_of_machines}")
            print(instance)

        return OSSP_problem(instance)

    def get_dim(self):
        """
        Get the total number of operations in the problem.
        
        Returns:
            int: Total number of operations.
        """
        return self.number_of_jobs * self.number_of_machines

    def get_num_machine_list(self):
        """
        Get the number of machines in the problem.
        
        Returns:
            int: Number of machines.
        """
        return self.number_of_machines

    def calculate_makespan(self, solution, verbose=False):
        """
        Calculate the makespan for a given solution.
        
        Args:
            solution (list): A sequence of operations representing the solution.
            verbose (bool): If True, print the schedule for each machine.
        
        Returns:
            int: The makespan time.
            list: The schedule for each machine.
        """
        schedule = [[] for _ in range(self.number_of_machines)]

        for operation in solution:
            this_job = operation // self.number_of_machines
            this_machine = operation % self.number_of_machines
            this_execution_time = self.data[this_job][this_machine]
            
            operation_executed = False
            
            while not operation_executed:
                job_already_executing_list = []
                
                for other_machine in range(self.number_of_machines):
                    if this_job in schedule[other_machine]:
                        job_already_executing_list.append(True)
                        
                        time_this_machine = len(schedule[this_machine])
                        time_other_machine = len(schedule[other_machine])

                        start_operation_time_this_machine = time_this_machine
                        finish_operation_time_other_machine = self.find_index(schedule[other_machine], this_job)

                        if ((time_this_machine <= time_other_machine) and 
                            (finish_operation_time_other_machine < start_operation_time_this_machine)) or \
                            (time_this_machine >= time_other_machine):
                            job_already_executing_list[other_machine] = False
                    else:
                        job_already_executing_list.append(False)
                
                job_already_executing = any(job_already_executing_list)

                if job_already_executing:
                    schedule[this_machine].append('-')  # Wait until the job finishes on the other machine
                else:
                    schedule[this_machine].extend([this_job] * this_execution_time)  # Execute the operation
                    operation_executed = True

        if verbose:
            for machine_schedule in schedule:
                print(machine_schedule)
        
        makespan = max(len(machine_schedule) for machine_schedule in schedule)
        return makespan, schedule

    def find_index(self, other_machine_schedule, job):
        
        """
        Find the last occurrence of a job in a machine's schedule.
        
        Args:
            other_machine_schedule (list): Schedule of operations for a machine.
            job (int): Job number to find.
        
        Returns:
            int: Index of the last occurrence of the job, or -1 if not found.
        """

        try:
            index = len(other_machine_schedule) - 1 - other_machine_schedule[::-1].index(job)
            return index
        except ValueError:
            return -1
        
    def generate_gantt_data(self, solution):
        """
        Generate data for Gantt chart visualization.
        
        Args:
            solution (list): A 2D list representing the scheduling solution.
        
        Returns:
            list: A list of dictionaries containing scheduling information for Gantt chart plotting.
        """
        gantt_data = []
        
        for machine_num, machine_schedule in enumerate(solution):
            if not machine_schedule:
                continue
            
            current_job = machine_schedule[0]
            start_time = 0
            duration = 1
            
            for time_slot in range(1, len(machine_schedule)):
                job = machine_schedule[time_slot]
                if job == current_job:
                    duration += 1
                else:
                    gantt_entry = self.get_dict(current_job, machine_num, start_time, duration)
                    gantt_data.append(gantt_entry)
                    
                    current_job = job
                    start_time = time_slot
                    duration = 1
            
            gantt_entry = self.get_dict(current_job, machine_num, start_time, duration)
            gantt_data.append(gantt_entry)
        
        cleaned_gantt_data = self.remove_idle_times(gantt_data)
        
        return cleaned_gantt_data

    def get_dict(self, job_num, machine_num, start, duration):
        """
        Create a dictionary with scheduling information for a single operation.
        
        Args:
            job_num (int/str): Job identifier.
            machine_num (int/str): Machine identifier.
            start (int): Start time of the operation.
            duration (int): Duration of the operation.
        
        Returns:
            dict: A dictionary with scheduling information.
        """
        return {
            'Job': f'job_{job_num}',
            'Machine': f'machine_{machine_num}',
            'Start': start,
            'Duration': duration,
            'Finish': start + duration
        }

    def remove_idle_times(self, scheduling):
        """
        Remove idle times from the scheduling data.
        
        Args:
            scheduling (list): List of scheduling dictionaries.
        
        Returns:
            list: Scheduling data without idle times.
        """
        return [sched for sched in scheduling if sched['Job'] != 'job_-']

    def plot_gantt_chart(self, results, save=False):
        """
        Plot a Gantt chart for the job shop scheduling problem.
        
        Args:
            results (list): List of dictionaries containing scheduling information.
            save (bool): If True, save the Gantt chart as a PNG file.
        """
        schedule = pd.DataFrame(results)

        jobs = sorted(schedule['Job'].unique())
        machines = sorted(schedule['Machine'].unique())
        makespan = schedule['Finish'].max()

        bar_style = {'alpha': 1.0, 'lw': 25, 'solid_capstyle': 'butt'}
        text_style = {'color': 'white', 'weight': 'bold', 'ha': 'center', 'va': 'center'}
        colors = mpl.cm.Dark2.colors

        schedule.sort_values(by=['Job', 'Start'], inplace=True)
        schedule.set_index(['Job', 'Machine'], inplace=True)

        fig, ax = plt.subplots(2, 1, figsize=(12, 5 + (len(jobs) + len(machines)) / 4))

        for jdx, job in enumerate(jobs, 1):
            for mdx, machine in enumerate(machines, 1):
                if (job, machine) in schedule.index:
                    start_time = schedule.loc[(job, machine), 'Start']
                    finish_time = schedule.loc[(job, machine), 'Finish']
                    ax[0].plot([start_time, finish_time], [jdx] * 2, c=colors[mdx % 7], **bar_style)
                    ax[0].text((start_time + finish_time) / 2, jdx, machine, **text_style)
                    ax[1].plot([start_time, finish_time], [mdx] * 2, c=colors[jdx % 7], **bar_style)
                    ax[1].text((start_time + finish_time) / 2, mdx, job, **text_style)

        ax[0].set_title('Jobs Schedule')
        ax[0].set_ylabel('Jobs')
        ax[1].set_title('Machines Schedule')
        ax[1].set_ylabel('Machines')

        for idx, entities in enumerate([jobs, machines]):
            ax[idx].set_ylim(0.5, len(entities) + 0.5)
            ax[idx].set_yticks(range(1, 1 + len(entities)))
            ax[idx].set_yticklabels(entities)
            ax[idx].text(makespan, ax[idx].get_ylim()[0] - 0.2, f"{makespan:.1f}", ha='center', va='top')
            ax[idx].plot([makespan] * 2, ax[idx].get_ylim(), 'r--')
            ax[idx].set_xlabel('Time')
            ax[idx].grid(True)

        fig.tight_layout()
        if save:
            file_path = script_dir / "results" / "execution_gantt.png"
            plt.savefig(file_path)
        plt.show()

class ACO:
    def __init__(self, problem, parameters: Dict[str, float], verbose: bool = False, save: bool = False):
        """
        Initialize the Ant Colony Optimization algorithm.
        
        Args:
            problem: The OSSP problem instance.
            parameters (dict): Dictionary of ACO parameters.
            verbose (bool): If True, print detailed information during execution.
            save (bool): If True, save results to file.
        """
        self.problem = problem
        self.ALPHA = parameters.get('alpha', 1.0)
        self.BETA = parameters.get('beta', 1.0)
        self.rho = parameters.get('rho', 0.5)
        self.tau0 = parameters.get('tau', 1.0)
        self.number_of_ants = parameters.get('number_of_ants', 10)
        self.number_of_generations = parameters.get('number_of_generations', 100)
        self.overall_best_solution_reward = self.rho * parameters.get('OverallـBestـSolutionـReward', 2/5)
        self.current_best_solution_reward = self.rho * parameters.get('Current_Best_Solution_Reward', 1/5)
        self.n_operations = problem.get_dim()
        self.verbose = verbose
        self.save = save
        self.num_machine_list = problem.get_num_machine_list()

        self.logger = self._setup_logging(verbose)

    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """
        Set up logging for the ACO algorithm.
        
        Args:
            verbose (bool): If True, set logging level to DEBUG.
        
        Returns:
            logging.Logger: Configured logger object.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def initialize_pheromone_and_ants(self):
        """
        Initialize the pheromone matrix and ant-related data structures.
        """
        self.pheromone_matrix = np.ones((self.n_operations + 1, self.n_operations)) * self.tau0
        self.best_solution = None
        self.best_solution_cost = float('inf')
        self.ant_solutions = [None] * self.number_of_ants
        self.ant_solution_costs = np.zeros(self.number_of_ants)

    def run_aco(self, seed: int = 0) -> Tuple[Optional[List[int]], float]:
        """
        Run the ACO algorithm.
        
        Args:
            seed (int): Random seed for reproducibility.
        
        Returns:
            tuple: The best solution found and its cost.
        """
        gantt_schedule_data_control = {}
        np.random.seed(seed)
        self.initialize_pheromone_and_ants()

        with alive_bar(self.number_of_generations, title="ACO Progress") as bar:
            for gen in range(1, self.number_of_generations + 1):
                current_cycle_times = [self._run_single_ant(ant_num) for ant_num in range(self.number_of_ants)]

                self.evaluate_solutions(gen)
                self.update_pheromone()

                if self.save:
                    gantt_schedule_data_control[gen] = [min(current_cycle_times), np.mean(current_cycle_times), max(current_cycle_times)]

                bar()

            if self.save:
                self._save_gantt_schedule_data(gantt_schedule_data_control)

        return self.best_solution, self.best_solution_cost

    def _run_single_ant(self, ant_num: int) -> float:
        """
        Generate a solution for a single ant and calculate its makespan.
        
        Args:
            ant_num (int): The index of the ant.
        
        Returns:
            float: The makespan of the ant's solution.
        """
        self.ant_solutions[ant_num] = self.generate_ant_solution()
        path_time, _ = self.problem.calculate_makespan(self.ant_solutions[ant_num])
        return path_time

    def generate_ant_solution(self) -> List[int]:
        """
        Generate a solution for an ant by walking through the problem's operations.

        Returns:
            list: A sequence of operations selected by the ant.
        """
        ant_solution = []
        remaining_operations = list(range(self.n_operations))
        current_operation_index = self.n_operations  # Start point for the ant

        for _ in range(self.n_operations):
            selected_operation_index = self.select_next_operation(current_operation_index, remaining_operations)
            ant_solution.append(selected_operation_index)
            current_operation_index = selected_operation_index
            remaining_operations.remove(selected_operation_index)
            
        return ant_solution

    def select_next_operation(self, current_operation_index: int, remaining_operations: List[int]) -> int:
        """
        Select the next operation for the ant to visit using a probability-based mechanism.

        Args:
            current_operation_index (int): The current operation index of the ant.
            remaining_operations (list): The list of operations that have not been visited.

        Returns:
            int: The index of the next operation to visit.
        """
        operation_selection_probabilities = self._calculate_operation_probabilities(current_operation_index, remaining_operations)
        next_node_index = np.random.choice(range(len(remaining_operations)), p=operation_selection_probabilities)
        return remaining_operations[next_node_index]

    def _calculate_operation_probabilities(self, current_operation_index: int, remaining_operations: List[int]) -> np.ndarray:
        """
        Calculate the selection probabilities for the remaining operations.

        Args:
            current_operation_index (int): The current operation index of the ant.
            remaining_operations (list): The list of operations that have not been visited.

        Returns:
            np.ndarray: The selection probabilities for each operation.
        """
        num_remaining_operations = len(remaining_operations)
        operation_selection_probabilities = np.zeros(num_remaining_operations)

        for i, operation in enumerate(remaining_operations):
            tau = self.pheromone_matrix[current_operation_index][operation]
            eta = 1 if current_operation_index == self.n_operations else 1 / self.problem.data[operation // self.num_machine_list][operation % self.num_machine_list]
            operation_selection_probabilities[i] = tau**self.ALPHA * eta**self.BETA

        normalized_probabilities = operation_selection_probabilities / operation_selection_probabilities.sum()
        return normalized_probabilities

    def evaporate(self):
        """Evaporate the pheromone across all paths, reducing their values by the evaporation rate."""
        self.pheromone_matrix *= (1 - self.rho)

    def evaluate_solutions(self, gen: int):
        """
        Evaluate all solutions generated by the ants in the current generation.

        Args:
            gen (int): The current generation number.
        """
        for i in range(self.number_of_ants):
            self.ant_solution_costs[i], _ = self.problem.calculate_makespan(self.ant_solutions[i])
        self.ibest = self.ant_solution_costs.argmin()

        if self.ant_solution_costs[self.ibest] < self.best_solution_cost:
            self.best_solution_cost = self.ant_solution_costs[self.ibest]
            self.best_solution = self.ant_solutions[self.ibest]

            self.logger.info(f"New best solution {self.best_solution_cost} at generation {gen}")

    def update_pheromone(self):
        """
        Update the pheromone matrix based on the best solutions found.
        """
        self.evaporate()
        self.reward(self.best_solution, self.current_best_solution_reward)
        self.reward(self.ant_solutions[self.ibest], self.overall_best_solution_reward)

    def reward(self, ant_solution: List[int], delta: float):
        """
        Reward the paths taken in a given solution by increasing the pheromone levels.

        Args:
            ant_solution (list): The solution path to reward.
            delta (float): The amount to increase the pheromone for the path.
        """
        current = self.n_operations
        for operation in ant_solution:
            self.pheromone_matrix[current][operation] += delta
            current = operation

    def _save_gantt_schedule_data(self, data: Dict[int, List[float]]):
        """
        Save Gantt schedule data to a file.

        Args:
            data (dict): The Gantt schedule data to save.
        """
        file_path = script_dir / "results/ACO_cycles_gantt_schedule_data.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
        self.logger.info(f"Gantt schedule data saved to {file_path}")



def main():


    try:
        # Load the problem instance
        problem = OSSP_problem.load_instance(instance, verbose)
        
        # Initialize the ACO algorithm
        aco = ACO(problem, parameters, verbose, save)
        
        start_time = time.time()

        # Run ACO algorithm
        solution_machine_schedules, solution_cost = aco.run_aco(seed=777)
        
        # End timing
        end_time = time.time()
        total_time = end_time - start_time

        print(f"Best Solution ever found using ACO: {solution_machine_schedules} {solution_cost}")
        print(f"\nTime needed to find solution with ACO: {total_time}")

    except Exception as e:
        print(f"An error occurred: {e}")

    return problem, solution_machine_schedules, solution_cost


def plot_cycles_data():

    if plot_cycles:
        try:
            file_path = script_dir / "results"/"ACO_cycles_gantt_schedule_data.json"             
            with open(file_path, 'w') as f:
                data = json.load(f)

            num_generations = len(data)
            generations = np.arange(1, num_generations + 1)
            gen_min = np.zeros(num_generations, dtype=np.uint)
            gen_mean = np.zeros(num_generations, dtype=np.uint)
            gen_max = np.zeros(num_generations, dtype=np.uint)

            for gen, values in data.items():
                index = int(gen) - 1
                gen_min[index], gen_mean[index], gen_max[index] = values


            df = pd.DataFrame({
                'Generation': generations,
                'Min': gen_min,
                'Mean': gen_mean,
                'Max': gen_max
            })


            plt.figure(figsize=(12, 8))
            plt.plot(df['Generation'], df['Min'], color='g', marker='o', linestyle='-', label='Min')
            plt.plot(df['Generation'], df['Mean'], color='y', marker='s', linestyle='-', label='Mean')
            plt.plot(df['Generation'], df['Max'], color='b', marker='^', linestyle='-', label='Max')

            plt.fill_between(df['Generation'], df['Min'], df['Max'], color='gray', alpha=0.2)  # Highlighting range

            plt.legend(loc="upper right")
            plt.title("ACO Generations Timeline Data")
            plt.xlabel("Number of Generations")
            plt.ylabel("Solution Cost")
            plt.grid(True)

            if save:
                file_path = script_dir / "results"/"ACO_cycles_results.png"
                plt.savefig(file_path)

            plt.show()

        except FileNotFoundError:
            print("File not found: ACO_cycles_gantt_schedule_data.json")
        except json.JSONDecodeError:
            print("Error decoding JSON from the file.")
        except Exception as e:
            print(f"An error occurred while plotting cycles: {e}")






def perform_local_search(problem, solution_machine_schedules, solution_cost):
    if do_local_search:
        try:
            start = time.time()
            print('Start solution optimization using LOCAL SEARCH')
            
            num_swap = max(len(solution_machine_schedules) // 5, 1)  # Ensure at least one swap
            x, fx = iterated_local_search(problem, num_tries=num_tries, num_swap=num_swap, init_sol=solution_machine_schedules)
            
            if fx < solution_cost:  # If local search improves solution
                solution_machine_schedules = x
                solution_cost = fx

            end = time.time()
            total_time = end - start
            print(f"\nTime needed to improve solution with Local Search: {total_time}")
            print(f"Best Solution ever found using Local Search on ACO Gantt Schedule Data: {x} {fx}")

        except Exception as e:
            print(f"An error occurred during local search: {e}")

    return solution_machine_schedules, solution_cost

def print_gantt_chart(problem, solution_machine_schedules):
    if plot_gantt_chart_gantt:
        try:
            _, machine_schedules = problem.calculate_makespan(solution_machine_schedules)
            gantt_schedule_data = problem.generate_gantt_data(machine_schedules)
            problem.plot_gantt_chart(gantt_schedule_data, save)
        
        except Exception as e:
            print(f"An error occurred while plotting Gantt chart: {e}")




if __name__ == "__main__":
    script_dir = Path(__file__).parent
    

    parameters = {
        'alpha': 1,
        'beta': 1,
        'rho': 0.1,
        'tau': 1,
        'Overall_Best_Solution_Reward': 2 / 3,
        'Current_Best_Solution_Reward': 1 / 3,
        'number_of_ants': 25,
        'number_of_generations': 20
    }


    instance = '44_1.txt'
    plot_gantt_chart_gantt = True
    plot_cycles = True
    do_local_search = True
    num_tries = 30
    save = True
    verbose = True

    print(parameters)
    print(instance)

    problem, solution_machine_schedules, solution_cost = main()
    plot_cycles_data()
    solution_machine_schedules, solution_cost = perform_local_search(problem, solution_machine_schedules, solution_cost)
    print_gantt_chart(problem, solution_machine_schedules)