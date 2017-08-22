"""
Base Estimator to build tcup
By Bowen Weng, Aug, 2017

"""

import pickle
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from tpot import TPOTClassifier, TPOTRegressor
from joblib import Parallel, delayed

class TCUPData():

	def __init__(self, features, target):
		self.features = features
		self.target = target

	def getSample(self, size):
		self.features, self.target = shuffle(self.features, self.target, random_state=42)
		f = self.features[0:size]
		t = self.target[0:size]
		return f, t

	def getCategoricalSample(self, size, category):
		self.features, self.target = shuffle(self.features, self.target, random_state=42)
		c = np.zeros(self.target.shape).fill(category)
		i = np.where(np.equal(c, self.target))[0:size]
		return self.features[i], self.target[i]


class TCUPBase(BaseEstimator):


	def __init__(self, n_base_estimator=20, n_estimator_pool=50, split_ratio=0.1, 
				 subsampling=0.6, n_jobs=-1, cv=3, verbosity=0,
				 tpot_generations=3, tpot_population_size=5, tpot_offspring_size=None,
				 tpot_mutation_rate=0.9, tpot_crossover_rate=0.1,
				 tpot_scoring=None, tpot_cv=5, tpot_subsample=1.0, tpot_n_jobs=1,
				 tpot_max_time_mins=None, tpot_max_eval_time_mins=5,
				 tpot_random_state=None, tpot_config_dict=None, tpot_warm_start=False,
				 tpot_verbosity=0, tpot_disable_update_check=False):
		self.n_base_estimator = n_base_estimator
		self.n_estimator_pool = n_estimator_pool
		self.split_ratio = split_ratio
		self.subsampling = subsampling
		self.n_jobs = n_jobs
		self.cv = cv
		self.verbosity = verbosity

		self.tpot_generations = tpot_generations
		self.tpot_population_size = tpot_population_size
		self.tpot_offspring_size = tpot_offspring_size
		self.tpot_mutation_rate = tpot_mutation_rate
		self.tpot_crossover_rate = tpot_crossover_rate
		self.tpot_scoring = tpot_scoring
		self.tpot_cv = tpot_cv
		self.tpot_subsample = tpot_subsample
		self.tpot_n_jobs = tpot_n_jobs
		self.tpot_max_time_mins = tpot_max_time_mins
		self.tpot_max_eval_time_mins = tpot_max_eval_time_mins
		self.tpot_random_state = tpot_random_state
		self.tpot_config_dict = tpot_config_dict
		self.tpot_warm_start = tpot_warm_start
		self.tpot_verbosity = tpot_verbosity
		self.tpot_disable_update_check = tpot_disable_update_check
		self.tpot_params = {"generations":tpot_generations, 
							"population_size":tpot_population_size, 
							"offspring_size":tpot_offspring_size, 
							"mutation_rate":tpot_mutation_rate, 
							"crossover_rate":tpot_crossover_rate, 
							"scoring":tpot_scoring,
							"cv":tpot_cv, 
							"subsample":tpot_subsample, 
							"n_jobs":tpot_n_jobs,
							"max_time_mins":tpot_max_time_mins, 
							"max_eval_time_mins":tpot_max_eval_time_mins,
							"random_state":tpot_random_state, 
							"warm_start":tpot_warm_start,
							"verbosity":tpot_verbosity, 
							"disable_update_check":tpot_disable_update_check
							}


	def _tpot_classifier(self, params=None):
		# classifier = TPOTClassifier(generations=params["generations"],population_size=params["population_size"],offspring_size=params["offspring_size"],mutation_rate = params["mutation_rate"],crossover_rate = params["crossover_rate"],scoring = params["scoring"],cv = params["cv"],subsample = params["subsample"],n_jobs = params["n_jobs"],max_time_mins = params["max_time_mins"],max_eval_time_mins = params["max_eval_time_mins"],random_state= params["random_state"],warm_start = params["warm_start"],verbosity = params["verbosity"],disable_update_check = params["disable_update_check"])
		classifier = TPOTClassifier(generations=params["generations"], population_size=params["population_size"], cv=5, n_jobs=-1,
									max_eval_time_mins = params["max_eval_time_mins"], random_state=42, warm_start = params["warm_start"], verbosity=params["verbosity"])
		return classifier

	def _tpot_regressor(self, params=None):
		regressor = TPOTRegressor()
		regressor.set_params(**params)
		return regressor 

	def _tpot_cvp(self, estimator, features, target):
		predicts = cvp(estimator, features, target, cv=self.cv, n_jobs=self.tpot_n_jobs)
		return predicts

	def fit(self, features, target):
		split_size = int(features.shape[0] * self.split_ratio)
		datagen = TCUPData(features, target)

		estimators = []
		estimators_score = []
		failure_features = []
		failure_target = []
		while len(estimators) < self.n_estimator_pool:	# keep split estimation
			print ("|","#"*(len(estimators)+1), end="\r")
			# build estimator
			if self.classification:
				tpot_estimator = self._tpot_classifier(params=self.tpot_params)
			if self.regression:
				raise NotImplementedError

			# bootstrap
			if len(failure_target) < split_size:
				sample_features, sample_target = datagen.getSample(split_size)
			else:
				sample_features = np.array(failure_features)
				sample_target = np.array(failure_target)
				failure_features = []
				failure_target = []
			# fit
			tpot_estimator.fit(sample_features, sample_target)
			tpot_steps = tpot_estimator.fitted_pipeline_.steps
			best_est = Pipeline(tpot_steps)
			estimators += [best_est]
			predicts = self._tpot_cvp(best_est, sample_features, sample_target)
			failure_index = np.where(np.not_equal(predicts, sample_target))[0]
			estimators_score += [1-failure_index.shape[0]/sample_target.shape[0]]
			failure_features += sample_features[failure_index].tolist()
			failure_target += sample_target[failure_index].tolist()
			
		estimators = np.array(estimators)
		self.base_estimators = np.concatenate((estimators[np.argsort(estimators_score[::-1])[0:int(self.n_base_estimator)//2]], estimators[np.argsort(estimators_score)[0:int(self.n_base_estimator)//2]]),axis=0)
		print (self.base_estimators.shape)
		F_list = []
		for i in range(self.n_base_estimator):
			print ("|","#"*(i+1), end="\r")
			self.base_estimators[i].fit(features,target)
			if self.classification:
				F_list += [np.expand_dims(cvp(self.base_estimators[i], features, target, cv=self.cv, n_jobs=self.n_jobs), axis=1)]
			if self.regression:
				raise NotImplementedError
		F = np.concatenate(F_list, axis=1)
		if self.classification:
			self.top_estimator = self._tpot_classifier(params = self.tpot_params)
		if self.regression:
			raise NotImplementedError
		self.top_estimator.fit(F, target)

	def predict(self, features):
		F_list = []
		for i in range(self.n_base_estimator):
			if self.classification:
				F_list += [np.expand_dims(self.base_estimators[i].predict(features), axis=1)]
			if self.regression:
				raise NotImplementedError
		F = np.concatenate(F_list, axis=1)
		predictions = self.top_estimator.predict(F)
		return predictions

	def score(self, features, target):
		predictions = self.predict(features)
		if self.classification:
			score = accuracy_score(target, predictions)
		if self.regression:
			score = r2_score(target, predictions)
		return score

	def save(self, path):
		state = self.__dict__.copy
		with open(path, "wb") as p:
			pickle.dump(state, p)

	def load(self, path):
		with open(path, "rb") as p:
			state = pickle.load(p)
		self.__dict__.update(state())

				

if __name__ == "__main__":
	pass