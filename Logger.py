import os
import subprocess
import time
import pandas as pd
from eckity.algorithms.simple_evolution import SimpleEvolution
import psutil

class Logger():
    def __init__(self, columns: dict = None, dump_every: int = 100, output_path: str = "logs.csv"):
        self._columns = columns if columns is not None else {}
        self._log_data = []  # List to hold log entries
        self._dump_every = dump_every
        self._output_path = output_path if output_path is not None else "logs.csv"
        self._first = True

    def update_column(self, name: str, lamd: callable):
        '''
        :param name: name of column in logger
        :param lamd: lambda, will be probed with each log
        '''
        self._columns[name] = lamd

    def log(self, *args, **kwargs):
        log_entry = {key: str(self._columns[key]()) for key in self._columns}
        self._log_data.append(log_entry)
        if len(self._log_data) >= self._dump_every:
            if self._first and os.path.exists(self._output_path):
                with open(self._output_path, "a") as file:
                    file.write('###\n')
            if self._first:
                self.log_headers(self._output_path)
            self.to_csv(self._output_path, append=True, header=False)
            self.empty_logs()
            self._first = False

    def add_gen_col(self, algo: SimpleEvolution = None):
        self.update_column("gen", (lambda: algo.event_name_to_data('')['generation_num']) if algo is not None else (lambda: None))

    def add_time_col(self):
        self.update_column("time", lambda: time.time())
    
    def add_str_col(self, name: str, value: str):
        self.update_column(name, lambda: value)
        
    def add_cpu_measure_col(self):
        self.update_column("measure", lambda: 0)
        self.update_column("type", lambda: "CPU")

    def add_slurm_cpu_measure_col(self, job_id: str):
        self.update_column("measure", lambda: str(subprocess.check_output(["sstat", "-j" + job_id, "-a", "--format=ConsumedEnergyRaw"])).split("\\n")[-2].strip())
        self.add_str_col("type", "CPU")
    
    def add_gpu_measure_col(self):
        self.update_column("measure", lambda: str(subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv"]))[18:-5])    
        self.add_str_col("type", "GPU")
        
    def add_best_of_gen_col(self, algo: SimpleEvolution):
        #self.update_column("best_of_gen", lambda: algo.get_individual_evaluator().evaluate_individual(algo.best_of_gen) if algo.best_of_gen is not None else 0)
        self.update_column("best_of_gen", lambda: algo.best_of_gen.fitness.get_pure_fitness() if algo.best_of_gen.fitness is not None else 0)
    def add_average_col(self, algo: SimpleEvolution):
        self.update_column("average", lambda: float(algo.get_average_fitness()[0]))
        
    def to_csv(self, path: str, append: bool = False, header: bool = False):
        df = pd.DataFrame(self._log_data)
        if append:
            df.to_csv(path, index=False, mode='a', header=header)
        else:
            df.to_csv(path, index=False, header=header)

    def empty_logs(self):
        self._log_data.clear()
    
    def log_headers(self, path: str):
        headers = list(self._columns.keys())
        with open(path, "a") as file_obj:
            file_obj.write(f"{','.join(headers)}\n")

    def num_logs(self):
        return len(self._log_data)
    
    def get_df(self):
        return pd.DataFrame(self._log_data)
    
    def add_memory_col(self, units: str, process: psutil.Process | None = None):
        if process is None:
            process = psutil.Process()

        units_dict = {'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3}
        self.update_column("MEMORY", lambda: process.memory_info().rss / units_dict[units] if units in units_dict else 1024)
