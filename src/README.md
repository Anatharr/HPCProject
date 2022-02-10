# HPCProject

## Installation
### Linux
Please don't forget to reboot after installation

### Windows
Install latest version of [Perl](https://www.perl.org/get.html), add it to your path and then install **openssl** headers with :
```powershell
    cpan install Net::SSL
```
Please take note of your Perl installation folder.

## Usage

### Compile the sources
#### Linux
```bash
    nvcc ../src/multiattack.cu -o attack -lssl -lcrypto
```
#### Windows
```powershell
    $PERL_FOLDER = "C:\Programmes\Perl"     # Path to installation folder of Perl
    nvcc ../src/multiattack.cu -o attack -I $PERL_FOLDER'\c\include'
```

### Run the binary
```
    ./attack 1 1 ../src/wordlists/dict.txt ../src/hash_db/shadowSmall.txt
```