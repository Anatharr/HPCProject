all: managed unmanaged

managed:
	nvcc ./src/multiattackManaged.cu -o ./bin/attackManaged

unmanaged:
	nvcc ./src/multiattackUnmanaged.cu -o ./bin/attackUnmanaged

optimized:
	nvcc ./src/multiattackOptimized.cu -o ./bin/attackOptimized

run:
	./bin/attack ./src/wordlists/dict_sha.txt ./src/hash_db/shadowSmall.txt 

clean: 
	rm ./bin/attack