#!/usr/bin/python3

import os, platform, numpy as np

DISABLE_PBAR = True

def run_benchmark(n_proc_max=100, modes = ["Managed"]):

    for mode in modes:
        for n_proc in range(1, n_proc_max, 10):
            COMMAND_NAME = f"./bin/attack{mode}" if platform.system() == 'Linux' else f".\\bin\\attack{mode}.exe"

            print(f"[+] Running multiattack with mode={mode} and n_proc={n_proc}")
            os.system(f"{COMMAND_NAME} ./src/wordlists/dict_sha.txt ./src/hash_db/shadow.txt {n_proc} {int(DISABLE_PBAR)}")

if __name__ == '__main__':
    run_benchmark(3000, ["Managed", "Unmanaged"])