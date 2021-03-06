all: managed unmanaged optimized

managed:
	nvcc ./src/multiattackManaged.cu -o ./bin/attackManaged

unmanaged:
	nvcc ./src/multiattackUnmanaged.cu -o ./bin/attackUnmanaged

optimized:
	nvcc ./src/multiattackOptimized.cu -o ./bin/attackOptimized

run_managed:
	./bin/attackManaged ./src/wordlists/dict_sha.txt ./src/hash_db/shadow.txt 

run_unmanaged:
	./bin/attackUnmanaged ./src/wordlists/dict_sha.txt ./src/hash_db/shadow.txt 

run_optimized:
	./bin/attackOptimized ./src/wordlists/dict_sha.txt ./src/hash_db/shadow.txt 

benchmark: 
	python ./src/benchmark.py

clean: 
	rm ./bin/attack