#! /usr/bin/env python3


import logging

import psutil
from my_utils import os as mos


class Logger(object):
    def __init__(self, job_name, file_path, log_level=logging.INFO, mode="w"):
        self.__logger = logging.getLogger(job_name)
        self.__logger.setLevel(log_level)
        self.__fh = logging.FileHandler(filename=file_path, mode=mode)
        self.__formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.__fh.setFormatter(self.__formatter)
        self.__logger.addHandler(self.__fh)

    @property
    def logger(self):
        return self.__logger


logger = Logger(job_name="sgd_profiler", file_path="./logs/cpu_usage.log").logger

key_str = "train_cifar_sgd.py"
cmd = f"ps aux | grep '{key_str}' | grep -v grep | awk '{{print $2}}'"
pid = mos.run_cmd(cmd)
if "\n" in pid:
    pid = pid.split("\n")[0]

p = psutil.Process(pid=int(pid))
while True:
    logger.info(p.cpu_percent(0.02))
