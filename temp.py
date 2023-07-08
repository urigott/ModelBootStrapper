from lightgbm import LGBMClassifier as LGBMClassifier
import ModelBootStrapper as mbs
import pandas as pd
import numpy as np

X = pd.DataFrame(np.random.normal(size=(100, 3)), columns=list("ABC"))
y = pd.Series(np.random.binomial(1, 0.75, size=100))

m = mbs.ModelBootStrapper(LGBMClassifier(), n_boot=5)
m.fit(X, y)

preds = m.predict(X, sort_estimations=True)

ppv = m.calculate_ppv(X, y)
print(type(ppv))
