#!/usr/bin/env python2
"""Demo for Experiment class."""

from experiment import Experiment

class MultiProcessing(Experiment):

    def __init__(self, kernel):
        super().__init__()
        self._params['Trial'] = list(range(1000))

    def task(configuration):
        trial = configuration[0]

        # do your stuff

        return results


    def result(retval):
        self.results.append(retval)



if __name__=='__main__':
    exp = MultiProcessing('linear')
    exp.run(nproc=16)
    exp = MultiProcessing('poly')
    exp.run(nproc=16)
    exp = MultiProcessing('linear')
    exp.run(nproc=16)
        
