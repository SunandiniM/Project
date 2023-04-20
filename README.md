# CSC 591-021 ASE Group-3 Project - Spring 23
[![DOI](https://zenodo.org/badge/588367919.svg)](https://zenodo.org/badge/latestdoi/588367919)
[![Collaborators](https://img.shields.io/badge/Collaborators-3-purple.svg?style=flat)](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Project/graphs/contributors)
[![Language](https://img.shields.io/badge/Language-Python-orange.svg?style=flat)](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Project/search?l=python)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Project/blob/main/LICENSE)
[![Tests](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Project/actions/workflows/tests.yaml/badge.svg)](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Project/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/NCSU-CSC-591-021-Spring-23-Group-3/Homeworks/branch/main/graph/badge.svg)](https://codecov.io/gh/NCSU-CSC-591-021-Spring-23-Group-3/Project/branch/main)

# About
This repo is an extension of the [Homeworks](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Homeworks) repo and contains the final project of CSC 591 021 Spring 23 <b>Group 3</b>.

# Steps to run
  1. Install Python 3.10.6
  2. Run ```pip install -r requirements.txt```
  3. cd into src folder and run ```python main.py -f <csv_file_path>```. Use ```--help``` option for a list of all options and change arguments likewise. The tests of HW6 are still functional and project code can be verified by passing 'all' or a test case name with the -g option.
  4. Examples:<br/>
  Run project for default file of *auto2.csv* -  ```python main.py```<br/>
  Run project for custom file of *auto93.csv* - ```python main.py -f ../etc/data/auto93.csv```<br/>
  Run project for custom file and run test cases too - ```python main.py -f ../etc/data/auto2.csv -g all```<br/>
  
# Files
  1. The <b>Report</b> is present at [/docs/Report.pdf](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Project/tree/main/docs/Report.pdf) folder
  2. A list of usage options and other settings are present in [config.py](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Project/blob/main/src/config.py)
  3. Data files are present at [/etc/data](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Project/tree/main/etc/data) folder
  4. After execution, outputs are automatically stored in [/etc/out](https://github.com/NCSU-CSC-591-021-Spring-23-Group-3/Project/tree/main/etc/out) folder

# Team Members
 - Kaushik Jadhav
 - Ajith Kumar Vinayakamoorthy Patchaimayil
 - Sunandini Medisetti 
