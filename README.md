# tcup
tcup is an ensemble learning framework that's built upon tpot (https://github.com/rhiever/tpot) to enhance performance with large-size high-dimensional data set.

### installing
```bash
pip install tpot
pip install deap
```
### example-classification
```python
from tpot import TPOTClassifier
from tcup import TCUPClassifier
import numpy as np
from sklearn.datasets import *
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.75, test_size=0.25)

est = TCUPClassifier(n_base_estimator=14, n_estimator_pool=30, split_ratio=0.1, 
                     subsampling=0.6, n_jobs=-1, cv=3, verbosity=0,
                     tpot_generations=3, tpot_population_size=10, tpot_offspring_size=None,
                 	   tpot_mutation_rate=0.9, tpot_crossover_rate=0.1,
                 	   tpot_scoring=None, tpot_cv=5, tpot_subsample=1.0, tpot_n_jobs=-1,
                 	   tpot_max_time_mins=10, tpot_max_eval_time_mins=2,
                 	   tpot_random_state=None, tpot_config_dict=None, tpot_warm_start=True,
                 	   tpot_verbosity=0, tpot_disable_update_check=False)
est.fit(X_train, y_train)
score = est.score(X_test, y_test)
print ("tcup score: ", score)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print("tpot score: ", tpot.score(X_test, y_test))
```        
