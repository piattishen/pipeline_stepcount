# pipeline_stepcount

Python required package: pandas, datetime, os, subprocess, sys

## Oxford stepcount Install: https://github.com/OxWearables/stepcount?tab=readme-ov-file#install

Minimum requirements: Python>=3.9, Java 8 (1.8)

### The following instructions make use of Anaconda to meet the minimum requirements in Linux or Window:

1. Download & install Anaconda.

2. Once installed, launch the Anaconda Prompt.

3. Create a virtual environment:
```
$ conda create -n stepcount python=3.9 openjdk pip
```
This creates a virtual environment called stepcount with Python version 3.9, OpenJDK, and Pip.

5. Activate the environment:
```
$ conda activate stepcount
```
You should now see (stepcount) written in front of your prompt.

7. Install stepcount:
```
$ pip install stepcount
```
You are all set! The next time that you want to use stepcount, open the Anaconda Prompt and activate the environment (step 4). If you see (stepcount) in front of your prompt, you are ready to go!

## For linux (pipeline_linux_stepcount.py)
make sure you have
```
init conda run 
```
then run
```
$ activate stepcount
```
then you will see (stepcount) in front of your prompt

use cd /mnt/your-path to get into all the file address in your prompt.Input: python pipeline_oxford_stepcount.py <address_text_file> <output_directory
Example: python pipeline_oxford_stepcount.py DS_10_address.txt DS_10_test

address_text_file should be one address per line, with or without "". The file is Text format
Example: 
```
"/mnt/c/Users/piatt/mHealthLab/stepcount_algorithm_repeat/DS_10/DS_10-Free-LeftWrist.csv"
"/mnt/c/Users/piatt/mHealthLab/stepcount_algorithm_repeat/DS_10/DS_10-Free-RightWrist.csv"
```
output_directory should be the file folder location


## For window (pipeline_oxford_stepcount.py)
make sure you have
```
init conda run 
```
In the directory of pipeline_oxford_stepcount input cmd then Enter. In the command prompt, Input: python pipeline_oxford_stepcount.py <address_text_file> <output_directory
Example: python pipeline_oxford_stepcount.py DS_10_address.txt DS_10_test

address_text_file should be one address per line, with or without "". The file is Text format
Example: 
```
"C:\Users\piatt\mHealthLab\stepcount_algorithm_repeat\DS_10\DS_10-Free-LeftWrist.csv"
"C:\Users\piatt\mHealthLab\stepcount_algorithm_repeat\DS_10\DS_10-Free-RightWrist.csv"
```
output_directory should be the file folder location

The result of the Oxford stepcount algorithm will be saved in a file folder called "outputs" in the output_directory

output example:

file folder under outputs:

![image](https://github.com/user-attachments/assets/33dfe800-7181-44e0-80d4-ed73fd61fa0f)

all file for one dataset:

![image](https://github.com/user-attachments/assets/adb5bfed-59b7-4c2c-bcf8-c9c56288e370)

