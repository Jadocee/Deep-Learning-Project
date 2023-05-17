# Packaged with - dataset.py, resnet.py, train.py
# Author - Thomas Bandy (c3374048)

from train import Train


test = Train()

test.prepare_data()
test.print_checks()

test.begin_training(5)

#test.diagram()