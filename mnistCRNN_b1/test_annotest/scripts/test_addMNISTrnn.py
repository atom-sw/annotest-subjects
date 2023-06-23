from __future__ import print_function
import unittest


class test_ImportModule(unittest.TestCase):

    def test_import_module(self):
        import scripts.addMNISTrnn
        x = scripts.addMNISTrnn
