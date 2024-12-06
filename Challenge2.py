import numpy as np
import pandas as pd
import os
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.feature_selection import SelectFromModel

geno_data = pd.read_csv('train.genotype.txt', sep = ' ', header = None)
pheno_data = pd.read_csv('train.phenotype.txt', sep=' ', header = None)
geno_test = pd.read_csv('test.genotype.txt', sep=' ', header = None)

model = Lasso(alpha = 0.01, random_state=0)
model.fit(geno_data, pheno_data)

causal = SelectFromModel(model, threshold = 'median', max_features = 100)
causal.fit(geno_data, pheno_data)

geno_updated = causal.transform(geno_data)

updated_test = causal.transform(geno_test)

model2 = Lasso(alpha = 0.01, random_state = 0)
model2.fit(geno_updated, pheno_data)
predictions = model2.predict(updated_test)

pd.DataFrame(predictions).to_csv(f"predictions.csv", sep = " ", header = None, index = None)
os.system("zip -r predictions.zip predictions.csv")
