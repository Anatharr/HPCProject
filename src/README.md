# HPCProject

## Usage

### Compile the sources
    nvcc ../src/multiattack.cu -o attack -lssl -lcrypto

### Run the binary
    ./attack 1 1 ../src/wordlists/dict.txt ../src/hash_db/shadowSmall.txt