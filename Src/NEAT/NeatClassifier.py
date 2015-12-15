from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from neat import config, population, chromosome, genome, visualize
from neat.nn import nn_pure as nn
import math
import fileinput
from operator import itemgetter
import random

class NeatClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,**kwargs):
        self.network = None
        self.max_generations = kwargs.pop('max_generations',10)

    def get_params(self,deep=True):
        return {"max_generations" : self.max_generations}

    def fit(self,data,target):

        target = [t if t==0 else 1 for t in target]
        self.x = list(set(target))

        #for line in fileinput.input(self.config_file,inplace=1):
        #    print (line if not line.startswith("input_nodes") else "input_nodes         = %d" % len(data[0])).strip()

        #config.load(self.config_file)
        config.load(len(data[0]))
        chromosome.node_gene_type = genome.NodeGene

        def eval_fitness(population):
            for chromo in population:
                net = nn.create_ffphenotype(chromo)

                error = 0.0
                for i, inputs in enumerate(data):
                    net.flush() # not strictly necessary in feedforward nets
                    output = net.sactivate(inputs) # serial activation
                    error += (output[0] - target[i])**2
                chromo.fitness = 1 - math.sqrt(error/len(target))

        population.Population.evaluate = eval_fitness
        pop = population.Population()
        pop.epoch(self.max_generations, report=True, save_best=False)
        winner = pop.stats[0][-1]

        self.network = winner


    def predict(self, data):
        if(not self.network):
            raise Exception("Classifier must be fit before it can be used for predictions")

        brain = nn.create_ffphenotype(self.network)
        predictions = []
        for i, inputs in enumerate(data):
            output = brain.sactivate(inputs) # serial activation
            predictions.append(output[0])
        return [self.discretize_prediction(prediction) for prediction in predictions]

    def discretize_prediction(self,prediction):
        p = min(enumerate([math.fabs(cls - prediction) for cls in self.x]), key=itemgetter(1))[0]
        return (p if p==0 else 2)

#if __name__ == "__main__":
#    clf = NeatClassifier(max_generations=300)
#    input = [[random.random() for x in xrange(0,3)] for y in xrange(0,4)]
#    output = [0, 2, 2, 0]
#
#    clf.fit(input,output)
#    print(clf.predict(input))
