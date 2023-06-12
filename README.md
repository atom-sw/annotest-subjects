# aNNoTest subjects

The current repository contains 64 bugs collected by Islam et al. (2019),
who provided these bugs as links to
line numbers within different commits, each representing one bug.

Proposing aNNoTest, an annotation-based test generation tool 
for neural network programs, we managed to generate failing 
test cases that reproduce these bugs. 
The details of aNNoTest and how we generated
these failing test cases can be found in the 
following paper:

[An annotation-based approach for 
finding bugs in neural network programs, by
Mohammad Rezaalipour and Carlo A. Furia.](https://doi.org/10.1016/j.jss.2023.111669)

## Bugs information

File [bugs_info.csv](bugs_info.csv) contains information about these
64 bugs in this repository. For each bug, this file includes 
its project name, the framework/library used to develop it, and a link
to commit line numbers representing the bug.

For each bug, this repo includes the bash scripts to create
a Python environment with all of the bug's dependencies,
the dataset and files needed by that bug, and a failing test
that simply uses the inputs aNNoTest generated to reproduce that
bug.

## Reproducing bugs

To reproduce the bugs, follow the instructions below:

1. Install Anaconda 3 on your machine, following
the instructions on 
[Anaconda's documentation 
website](https://docs.anaconda.com/free/anaconda/install/index.html).


2. Set the enviroment varaible `ANACONDA3_DIRECTORY` to
point to `anaconda3` directory on your machine.
For instance, if anaconda3 is in your home directory
(i.e., `~/anaconda3`), run the following command
or put it in your `~/.profile` file.

```
export ANACONDA3_DIRECTORY=~/anaconda3
```

3. Clone the repository.

```
git clone git@github.com:mohrez86/annotest-subjects.git

cd annotest-subjects-dev
git checkout dev
```

4. For each bug, there is a directory in the repo.
First, `cd` to that directory (e.g. `densenet_b1`), and then, make the
bash scripts in that directory executable.

```
cd densenet_b1
chmod +x *.sh
```

5. Make a conda environment for the bug including
all the bug's dependencies.

```
./make_env.sh
```

6. Some bugs require some datasets or files.
for such bugs, there is a bash script named `make_data.sh`,
running which downloads these dependencies and puts them
in directory `~/annotest_subjects_data` on your machine.

```
./make_data.sh
```

7. To reproduce the bug and see the output result,
run the following command:
```
./run_manual_failing_test.sh
```


## Citations

[aNNoTest's Journal 
Paper:](https://www.sciencedirect.com/science/article/pii/S016412122300064X?via%3Dihub)

```
@article{Rezaalipour:2023,
title = {An annotation-based approach for finding bugs in neural network programs},
journal = {Journal of Systems and Software},
volume = {201},
pages = {111669},
year = {2023},
issn = {0164-1212},
doi = {https://doi.org/10.1016/j.jss.2023.111669},
url = {https://www.sciencedirect.com/science/article/pii/S016412122300064X},
author = {Mohammad Rezaalipour and Carlo A. Furia},
keywords = {Test generation, Neural networks, Debugging, Python},
}
```