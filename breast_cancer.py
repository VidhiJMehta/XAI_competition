from foldrpp import *
from timeit import default_timer as timer 
from datetime import timedelta

# Kaggle link: https://www.kaggle.com/yasserh/breast-cancer-dataset?select=breast-cancer.csv

attrs = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']
nums = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']
model = Classifier(attrs=attrs, numeric=nums, label='diagnosis', pos='B')

data = model.load_data('data/breast-cancer.csv')
data_train, data_test = split_data(data, ratio=0.9, rand=True)

X_train, Y_train = split_xy(data_train)
X_test,  Y_test = split_xy(data_test)

start = timer()
model.fit(X_train, Y_train, ratio=0.5)
end = timer()
model.print_asp()

Y_test_hat = model.predict(X_test)
acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')

# output
# diagnosis(X,'b') :- concave_points_worst(X,N27), N27=<0.142, not ab3(X), not ab4(X), not ab6(X), not ab7(X), not ab8(X).
# diagnosis(X,'b') :- area_se(X,N13), N13=<21.2, texture_worst(X,N21), N21=<26.19, perimeter_worst(X,N22), N22=<112.5.
# diagnosis(X,'b') :- fractal_dimension_se(X,N19), N19>0.013.
# diagnosis(X,'b') :- compactness_mean(X,N5), N5>0.067, concavity_worst(X,N26), N26=<0.18.
# ab1(X) :- radius_mean(X,N0), N0=<16.02.
# ab2(X) :- concavity_worst(X,N26), N26=<0.188, not ab1(X).
# ab3(X) :- area_worst(X,N23), N23>947.9, not ab2(X).
# ab4(X) :- fractal_dimension_mean(X,N9), N9=<0.056, concave_points_worst(X,N27), N27>0.111.
# ab5(X) :- radius_mean(X,N0), N0=<12.65.
# ab6(X) :- texture_mean(X,N1), N1>15.7, compactness_se(X,N15), N15=<0.027, concave_points_worst(X,N27), N27>0.132, not ab5(X).
# ab7(X) :- compactness_mean(X,N5), N5=<0.059, radius_se(X,N10), N10>0.641.
# ab8(X) :- texture_worst(X,N21), N21>33.33, texture_worst(X,N21), N21=<33.37.
# % acc 0.8947 p 0.8947 r 0.9444 f1 0.9189
# % foldr++ costs:  0:00:00.378154
