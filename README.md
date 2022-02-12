# HPCProject

## Installation
### Linux
**Please don't forget to reboot after CUDA installation...**

### Windows
After installing CUDA, install latest version of [Perl](https://www.perl.org/get.html), add it to your path and then install **openssl** headers with :
```powershell
    cpan install Net::SSL
```
Please take note of your Perl installation folder as you will need to add it to the include path if you want to execute any of the generators.

## Usage

### Compile the sources & run the binary
#### Linux
To ensure that the sources are correctly compiled just run `make` at the root project.

#### Windows
To ensure that the sources are correctly compiled just run `nmake` at the root project.

### Run the binary
All binaries are in the bin/ folder, you can run them withour arguments to show usage.    

### TODO List

- [x] nice documentation (func docstring, comments)
- [x] clean pre-link directive
- [x] optimisation with shadow_size, `if index > shadow_size` -> do nothing
- [x] if a hash is already found, put him in memory to optimize the next guesses
- [x] `cudaMalloc` use, without the managed 
