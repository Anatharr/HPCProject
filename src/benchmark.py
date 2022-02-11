#!/usr/bin/python3

import os, platform, numpy as np

DISABLE_PBAR = True

def run_benchmark(ratios=list(np.linspace(0,1,10)), modes = ["Managed"]):

    for mode in modes:
        for ratio in ratios:
            COMMAND_NAME = f"./bin/attack{mode}" if platform.system() == 'Linux' else f".\\bin\\attack{mode}.exe"

            print(f"[+] Running multiattack with mode={mode} and ratio={ratio}")
            os.system(f"{COMMAND_NAME} ./src/wordlists/dict_sha.txt ./src/hash_db/shadow.txt {round(ratio, 2)} {int(DISABLE_PBAR)}")

if __name__ == '__main__':
    run_benchmark(list(np.linspace(0,1,20)), ["Managed", "Unmanaged"])