import argparse
import json
import random
import os, subprocess
from csv import DictWriter
import multiprocessing
import itertools
import matplotlib.pyplot as plt

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config_path",
        type=str,
        default="grid_search.json",
        help="Location of config file"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of processes to run in parallel"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Location of experiment logs and results"
    )

    parser.add_argument(
        "--grid_search_results_path",
        default="grid_results.csv",
        help="Where to save grid search results"
    )

    return parser

def get_experiment_list(config: dict) -> (list[dict]):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item, but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of dicts, each of which encapsulates one job.
        *Example: {learning_rate: 0.001 , batch_size: 16 ...}
    '''
    # TODO: Go through the tree of possible jobs and enumerate into a list of jobs
    jobs = []
    # Extract parameter names and their corresponding values
    param_names = config.keys()
    param_values = [config[param] for param in param_names]

    # Create combinations of all parameters
    for values in itertools.product(*param_values):
        job = dict(zip(param_names, values))
        if all(key in job for key in param_names):
            jobs.append(job)
        else:
            print(f"Generated job is missing keys: {job}")

    return jobs

def worker(args: argparse.Namespace, job_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue):
    '''
    Worker thread for each worker. Consumes all jobs and pushes results to done_queue.
    :args - command line args
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, params))


def launch_experiment(args: argparse.Namespace, experiment_config: dict) ->  dict:
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    :configs: flags to use for this model run. Will be fed into
    scripts/main.py

    returns: flags for this experiment as well as result metrics
    '''
    print("Launching experiment with config:", experiment_config)
    
    # TODO: Launch the experiment
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    
    #Create unique results path 
    param_string = "_".join(f"{key}_{value}" for key, value in experiment_config.items())
    # Construct the command to run the experiment
    command = [
        "python", "main.py",  # Path to the main script to run
        "--learning_rate", str(experiment_config["learning_rate"]),
        "--batch_size", str(experiment_config["batch_size"]),
        "--num_epochs", str(experiment_config["num_epochs"]),
        "--regularization_lambda", str(experiment_config["regularization_lambda"]), 
        "--results_path", param_string
    ]
    # TODO: Parse the results from the experiment and return them as a dict

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    result_json_path = os.path.join('logs/jsons', f"{param_string}.json")
    if result.returncode == 0:
        # Assuming main.py outputs metrics as JSON to results_path
        with open(result_json_path, 'r') as f:
            metrics = json.load(f)  # Load metrics from the JSON file
    else:
        metrics = {"error": result.stderr}
    
    metrics.update(experiment_config)  # Combine with experiment config
    return metrics

    


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> dict:
    print(args)
    config = json.load(open(args.config_path, "r"))
    print("Starting grid search with the following config:")
    print(config)

    # TODO: From config, generate a list of experiments to run
    experiments = get_experiment_list(config)
    random.shuffle(experiments)

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for exper in experiments:
        job_queue.put(exper)

    print("Launching dispatcher with {} experiments and {} workers".format(len(experiments), args.num_workers))

    # TODO: Define worker fn to launch an experiment as a separate process.

    workers = []
    for _ in range(args.num_workers):
        worker_process = multiprocessing.Process(target=worker, args=(args, job_queue, done_queue))
        workers.append(worker_process)
        worker_process.start()

    # Add sentinel values to signal workers to exit
    for _ in range(args.num_workers):
        job_queue.put(None)

    # Accumulate results into a list of dicts
    grid_search_results = []
    for _ in range(len(experiments)):
        grid_search_results.append(done_queue.get())

    # Wait for all workers to finish
    for w in workers:
        w.join()
    
    # Save results to CSV
    if grid_search_results:
        keys = grid_search_results[0].keys()
        with open(args.grid_search_results_path, 'w', newline='') as f:
            writer = DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(grid_search_results)

    print("Dispatcher finished")

    
if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)