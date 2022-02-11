all:
	nvcc ./src/multiattack.cu -o ./bin/attack

run:
	./bin/attack ./src/wordlists/dict_sha.txt ./src/hash_db/shadowSmall.txt 

clean: 
	rm ./bin/attack