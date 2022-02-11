# HPCProject

## Installation
### Linux
**Please don't forget to reboot after installation !**

### Windows
Install latest version of [Perl](https://www.perl.org/get.html), add it to your path and then install **openssl** headers with :
```powershell
    cpan install Net::SSL
```
Please take note of your Perl installation folder.

## Usage

### Compile the sources & run the binary
#### Linux

To ensure that the sources are correctly compiled and that the binary will get the right files input, just run `make` at the root project.

#### Windows
```powershell
    cd bin\
    $PERL_FOLDER = "C:\Programmes\Perl"     # Path to installation folder of Perl
    nvcc ../src/multiattack.cu -o attack -I $PERL_FOLDER'\c\include'
    .\attack.exe ..\src\wordlists\dict_sha.txt ..\src\hash_db\shadowSmall.txt
```

### Run the binary
    

### TODO

- [x] nice documentation (func docstring, comments)
- [x] clean pre-link directive 
- [x] optimisation  shadow_size, `if index > shadow_size` -> do nothing
- [x] if a hash is already found, put him in memory to optimize the next guesses
- [x] `cudaMalloc` use, without the managed 
