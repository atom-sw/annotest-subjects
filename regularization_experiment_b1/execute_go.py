
import sys
import subprocess
import multiprocessing as mp
# from Queue import Queue
from multiprocessing import Queue
from threading import Thread

def run(e, a, n, m, d, l, r, p, x, o):
    subprocess.call('THEANO_FLAGS=device=gpu,floatX=float32 python go.py -e {0} -a {1} -n {2} -m {3} -d {4} -l {5} -r {6} -p {7} -x {8} -o {9}'.format(e, a, n, m, d, l, r, p, x, o), shell=True)

epoch = [50]
aug = [False]
noise = [True, False]
maxout = [True, False]
dropout = [True, False]
l1 = [False]
l2 = [False]
maxpooling = [False]
deep = [False]
noise_rate = [0.01]

for e in epoch:
    for a in aug:
        for n in noise:
            for m in maxout:
                for d in dropout:
                    for one in l1:
                        for two in l2:
                            for p in maxpooling:
                                for x in deep:
                                    for o in noise_rate:
                                        run(e, a, n, m, d, one, two, p, x, o)
