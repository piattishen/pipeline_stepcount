# pipeline_stepcount

Python required package: pandas, datetime, os, subprocess, sys
Oxford stepcount Install: https://github.com/OxWearables/stepcount?tab=readme-ov-file#install

Minimum requirements: Python>=3.9, Java 8 (1.8)

The following instructions make use of Anaconda to meet the minimum requirements:

Download & install Anaconda.
(Windows) Once installed, launch the Anaconda Prompt.
Create a virtual environment:
$ conda create -n stepcount python=3.9 openjdk pip
This creates a virtual environment called stepcount with Python version 3.9, OpenJDK, and Pip.
Activate the environment:
$ conda activate stepcount
You should now see (stepcount) written in front of your prompt.
Install stepcount:
$ pip install stepcount
You are all set! The next time that you want to use stepcount, open the Anaconda Prompt and activate the environment (step 4). If you see (stepcount) in front of your prompt, you are ready to go!

In the directory of pipeline_oxford_stepcount, input "cmd" to replace the original address. Input: python pipeline_oxford_stepcount.py <address_text_file> <output_directory>

address_text_file should be one address per line, with or without ""
output_directory should be the file folder location

Before running this pipeline, make sure you have already init Anaconda in your environment

The result of the Oxford stepcount algorithm will be saved in a file folder called "outputs" in the output_directory
