from tcupbase import *


class TCUPClassifier(TCUPBase):
	classification = True
	regression = False

class TCUPRegressor(TCUPBase):
	classification = False
	regression = True