from foldrpp import *
from timeit import default_timer as timer 
from datetime import timedelta

# Kaggle link: https://www.kaggle.com/adityakadiwal/water-potability

attrs = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
nums = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
model = Classifier(attrs=attrs, numeric=nums, label='Potability', pos='0')

data = model.load_data('data/water_potability.csv')
data_train, data_test = split_data(data, ratio=0.8, rand=True)

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
# potability(X,'0') :- ph(X,N0), N0>4.633, ph(X,N0), N0=<6.321, hardness(X,N1), N1>98.772, hardness(X,N1), N1=<207.821, chloramines(X,N3), N3>4.824, not ab7(X).
# potability(X,'0') :- ph(X,N0), N0>7.809, chloramines(X,N3), N3=<7.923, not ab12(X), not ab16(X).
# potability(X,'0') :- ph(X,N0), N0=<4.633, not ab17(X).
# potability(X,'0') :- ph(X,N0), N0=<5.285, chloramines(X,N3), N3>6.198, sulfate(X,N4), N4=<314.987, not ab19(X).
# potability(X,'0') :- ph(X,N0), N0>4.789, ph(X,N0), N0=<5.285, chloramines(X,N3), N3>9.83.
# potability(X,'0') :- ph(X,N0), N0=<5.285, chloramines(X,N3), N3>4.555, chloramines(X,N3), N3=<4.731.
# potability(X,'0') :- trihalomethanes(X,null), ph(X,N0), N0=<5.285, hardness(X,N1), N1=<162.457.
# ab1(X) :- organic_carbon(X,N6), N6=<9.21.
# ab2(X) :- hardness(X,N1), N1=<136.891, sulfate(X,N4), N4=<416.495.
# ab3(X) :- trihalomethanes(X,N7), N7>86.308.
# ab4(X) :- ph(X,N0), N0>5.496, chloramines(X,N3), N3=<5.52.
# ab5(X) :- ph(X,N0), N0>5.23, sulfate(X,N4), N4>390.095, sulfate(X,N4), N4=<393.555.
# ab6(X) :- solids(X,N2), N2=<14807.268, sulfate(X,N4), N4=<384.822.
# ab7(X) :- sulfate(X,N4), N4>383.239, not ab1(X), not ab2(X), not ab3(X), not ab4(X), not ab5(X), not ab6(X).
# ab8(X) :- ph(X,N0), N0=<8.037, organic_carbon(X,N6), N6>11.943.
# ab9(X) :- chloramines(X,N3), N3=<3.016.
# ab10(X) :- turbidity(X,N8), N8=<3.038, not ab8(X), not ab9(X).
# ab11(X) :- trihalomethanes(X,N7), N7>74.985, turbidity(X,N8), N8>3.729, turbidity(X,N8), N8=<4.341.
# ab12(X) :- sulfate(X,N4), N4=<288.678, not ab10(X), not ab11(X).
# ab13(X) :- hardness(X,N1), N1>244.12.
# ab14(X) :- sulfate(X,N4), N4=<290.337.
# ab15(X) :- turbidity(X,N8), N8=<2.591.
# ab16(X) :- solids(X,N2), N2>29298.359, sulfate(X,N4), N4=<317.302, turbidity(X,N8), N8=<4.647, not ab13(X), not ab14(X), not ab15(X).
# ab17(X) :- ph(X,N0), N0=<0.227, sulfate(X,N4), N4=<316.553.
# ab18(X) :- ph(X,N0), N0>4.913.
# ab19(X) :- solids(X,N2), N2>24211.631, not ab18(X).
# % acc 0.5152 p 0.7173 r 0.4038 f1 0.5167
# % foldr++ costs:  0:00:01.433236
