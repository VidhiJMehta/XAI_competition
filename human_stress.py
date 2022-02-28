from foldrm import *
from timeit import default_timer as timer
from datetime import timedelta

# Link: https://www.kaggle.com/laavanya/stress-level-detection
attrs = ['Humidity', 'Temperature', 'Step', 'count']
nums = ['Humidity', 'Temperature', 'Step', 'count']
model = Classifier(attrs=attrs, numeric=nums, label='Stress Level')

data = model.load_data('data/Stress-Lysis.csv')
data_train, data_test = split_data(data, ratio=0.8)


start = timer()
model.fit(data_train, ratio=0.5)
end = timer()

model.print_asp(simple=True)
Y = [d[-1] for d in data_test]
Y_test_hat = model.predict(data_test)
acc = get_scores(Y_test_hat, data_test)
print('% acc', round(acc, 4), '# rules', len(model.crs))
acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))
print('% foldrm costs: ', timedelta(seconds=end - start), '\n')


# Output:
# stress_level(X,'1') :- humidity(X,N0), N0>15.0, N0=<22.89.
# stress_level(X,'2') :- humidity(X,N0), N0>15.0.
# stress_level(X,'0') :- humidity(X,N0), N0>10.0.
# stress_level(X,'0') :- humidity(X,N0), N0=<10.0.
# % acc 0.9975 # rules 4
# % acc 0.9975 macro p r f1 0.9975 0.9975 0.9975 # rules 4
# % foldrm costs:  0:00:00.094088