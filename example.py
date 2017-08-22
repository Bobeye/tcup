from tpot import TPOTClassifier
from tcup import TCUPClassifier
import numpy as np
from keras.datasets import mnist
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
import time

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)


config = {"generations":5, 
		"population_size":50, 
		"offspring_size":None, 
		"mutation_rate":0.8, 
		"crossover_rate":0.1, 
		"scoring":"accuracy",
		"cv":5, 
		"subsample":1.0, 
		"n_jobs":-1,
		"max_time_mins":None, 
		"max_eval_time_mins":5,
		"random_state":None, 
		"warm_start":False,
		"verbosity":2, 
		"disable_update_check":False
		}
est = TCUPClassifier(n_base_estimator=14, n_estimator_pool=30, split_ratio=0.1, 
				 	 subsampling=0.6, n_jobs=-1, cv=3, verbosity=0,
				 	 tpot_generations=3, tpot_population_size=10, tpot_offspring_size=None,
                 	 tpot_mutation_rate=0.9, tpot_crossover_rate=0.1,
                 	 tpot_scoring=None, tpot_cv=5, tpot_subsample=1.0, tpot_n_jobs=-1,
                 	 tpot_max_time_mins=10, tpot_max_eval_time_mins=2,
                 	 tpot_random_state=None, tpot_config_dict=None, tpot_warm_start=True,
                 	 tpot_verbosity=0, tpot_disable_update_check=False)

t1 = time.time()
est.fit(X_train, y_train)
score = est.score(X_test, y_test)
t2 = time.time()
print ("tcup score: ", score)
print (t2-t1)

t3 = time.time()
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
t4 = time.time()
print (t4-t3)
print(tpot.score(X_test, y_test))
