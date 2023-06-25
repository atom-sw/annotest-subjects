# 1. aNNoTest subjects

The current repository contains 62 bugs collected in [Islam et al. (2019)](https://dl.acm.org/doi/10.1145/3338906.3338955),
which provided these bugs as links to
line numbers within different commits, each representing one bug.

Proposing aNNoTest, an annotation-based test generation tool 
for neural network programs, we managed to generate failing 
test cases that reproduce these bugs. 
The details of aNNoTest and how we generated
these failing test cases can be found in the 
following paper:

[An annotation-based approach for 
finding bugs in neural network programs, by
Mohammad Rezaalipour and Carlo A. Furia.](#13-citations).

## 1.1 Bugs information

File [bugs_info.csv](bugs_info.csv) contains information about the
62 bugs in this repository. For each bug, this file includes 
its project name, the framework/library used to develop it, and a link
to commit line numbers in its original repository, representing the bug.

For each bug, we added the following items to make it easy for users to reproduce it:

- aN annotations (within the bug's source code) required by aNNoTest to generate tests (e.g., [annotations](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/examples/example_gan.py#L38-L39)).

- Comment `repo_bug` within the bug's source code, indicating bug's location (e.g., [bug location](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/examples/example_gan.py#L45)).

- File `requirements.txt` containing project dependencies (e.g., [requirements.txt](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/requirements.txt)).

- Script `make_env.sh` that creates a conda environment for the bug and installs all the bug's dependencies (e.g., [make_env.sh](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/make_env.sh)).

- Script `make_data.sh` that downloads the data required by the bug. Only bugs that need such data have this script (e.g., [make_data.sh](https://github.com/atom-sw/annotest-subjects/blob/main/car_recognition_b1/make_data.sh)).

- The failing test we generated for the bug, using aNNoTest. We refer to this test as aNNoTest test. Following the instructions in the current repository, users can generate this test themselves too, using aNNoTest (e.g., [aNNoTest test](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/test_annotest/examples/test_example_gan.py#L11-L18)).

- File `annotest_test_name.txt` including the name of aNNoTest test that can reproduce the bug (e.g., [annotest_test_name.txt](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/annotest_test_name.txt)).

- A failing test in Pytest format that we manually wrote according to aNNoTest test to make it easy for users to understand what the aNNoTest test does (e.g., [Pytest manual test](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/tests_manual/test_failing.py)).

- Script `run_annotest_failing_test.sh` that activates the bug's conda environment and runs the `aNNoTest` test (e.g., [run_annotest_failing_test.sh](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/run_annotest_failing_test.sh)).

- Script `run_manual_failing_test.sh` that activates the bug's conda environment and runs the manually produced failing test (e.g., [run_manual_failing_test.sh](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/run_manual_failing_test.sh)).

- File `expected_error_message.txt`, including the error message aNNoTest test and the manually produced test must report (e.g., [expected_error_message.txt](https://github.com/atom-sw/annotest-subjects/blob/main/keras_adversarial_b1/expected_error_message.txt)).
 

## 1.2 Reproducing bugs

To reproduce the bugs, follow the instructions below:

1. Install Anaconda 3 on your machine, following
the instructions on 
[Anaconda's documentation 
website](https://docs.anaconda.com/free/anaconda/install/index.html).


2. Set the enviroment varaible `ANACONDA3_DIRECTORY` to
point to `anaconda3` directory on your machine.
For instance, if `anaconda3` is in your home directory
(i.e., `~/anaconda3`), run the following command
or put it in your `~/.profile` file.

```
export ANACONDA3_DIRECTORY=~/anaconda3
```

3. Clone the repository, and cd to the repository directory on your machine.

```
git clone git@github.com:atom-sw/annotest-subjects.git
```

```
cd annotest-subjects
```

4. For each bug, there is a directory in the repo.
First, `cd` to that directory (e.g. `keras_adversarial_b1`), and then, make the
bash scripts in that directory executable.

```
cd keras_adversarial_b1
```

```
chmod +x *.sh
```

5. Run the following bash script to make a conda environment for the bug and install all the bug's dependencies in that environment.

```
./make_env.sh
```

6. Some bugs require some datasets or files.
For such bugs, there is a bash script named `make_data.sh`,
running which downloads these dependencies and puts them
in directory `~/annotest_subjects_data` on your machine.

```
./make_data.sh
```

7. To reproduce the bug, you must install aNNoTest inside the
conda environment created by running script `make_env.sh`.
The name of the conda environment is the same as the bug 
id (i.e., the bug's directory name), which is `keras_adversarial_b1` in our example.
So, first activate the bug's environment, and then
install aNNoTest.

```
conda activate keras_adversarial_b1
```

```
pip install annotest
```

8. For each bug, we have already generated aNNoTest tests.
A bug's aNNoTest test is in directory `test_annotest`.
First remove the existing aNNoTest tests, and then,
nun aNNoTest to generate them.
All of the projects in this repository are already
annotated. So, you do not need to annotate them, yourself.

```
rm -rf test_annotest
```

```
annotest .
```

9. Run the following bash script to execute the
Hypothesis test generated by aNNoTest. This bash script
simply activates the bug's conda environment and runs
the Hypothesis test generated by aNNoTest. You can find the
name of these tests in files named `annotest_test_name.txt` (e.g., [annotest_test_name.txt](keras_adversarial_b1/annotest_test_name.txt)).

```
./run_annotest_failing_test.sh
```

The error message must be the same as the one in file `expected_error_message.txt` in the bug's directory.

10. For every bug, we have included a Pytest
failing test that also reproduces the bug. We wrote
these tests manually according to the Hypothesis tests aNNoTest
generated. Execute the following command to run the manual
failing test.

```
./run_manual_failing_test.sh
```

The error message must be the same as the one in file `expected_error_message.txt` in the bug's directory.

## 1.3 Citations

[aNNoTest's Journal 
Paper:](https://doi.org/10.1016/j.jss.2023.111669)

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

# 2. Mirrors

The current repository is a public mirror of
our internal private repository.
We have two public mirrors, which are as follows:

- https://github.com/atom-sw/annotest-subjects
- https://github.com/mohrez86/annotest-subjects
