# HPCProject

## Usage

### Compile the sources
    nvcc ../src/multiattack.cu -o attack -lssl -lcrypto

### Run the binary
    ./attack 1 1 ../src/wordlists/dict.txt ../src/hash_db/shadowSmall.txt

### TODO

- [ ] nice documentation (func docstring, comments)
- [ ] correct use of naming case
- [ ] clean pre-link directive 