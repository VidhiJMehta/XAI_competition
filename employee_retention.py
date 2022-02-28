from foldrpp import *
from timeit import default_timer as timer 
from datetime import timedelta

# Kaggle link: https://www.kaggle.com/tejashvi14/employee-future-prediction

attrs = ['Education','JoiningYear','City','PaymentTier','Age','Gender','EverBenched','ExperienceInCurrentDomain']
nums = ['JoiningYear', 'PaymentTier','Age', 'EverBenched','ExperienceInCurrentDomain']
model = Classifier(attrs=attrs, numeric=nums, label='LeaveOrNot', pos='0')

data = model.load_data('data/employee.csv')
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
# leaveornot(X,'0') :- joiningyear(X,N1), N1=<2017.0, not ab9(X), not ab10(X), not ab14(X), not ab16(X), not ab19(X), not ab20(X), not ab24(X), not ab28(X).
# leaveornot(X,'0') :- joiningyear(X,N1), N1>2017.0, paymenttier(X,N3), N3=<1.0, not ab6(X).
# ab1(X) :- education(X,'masters').
# ab2(X) :- city(X,'bangalore'), paymenttier(X,N3), N3=<1.0, not ab1(X).
# ab3(X) :- not education(X,'bachelors'), joiningyear(X,N1), N1>2013.0.
# ab4(X) :- education(X,'bachelors'), paymenttier(X,N3), N3>1.0.
# ab5(X) :- not city(X,'pune'), not ab4(X).
# ab6(X) :- not education(X,'bachelors').
# ab7(X) :- gender(X,'male'), experienceincurrentdomain(X,N7), N7>3.0, not ab5(X), not ab6(X).
# ab8(X) :- city(X,'new_delhi'), not gender(X,'male'), paymenttier(X,N3), N3=<1.0, experienceincurrentdomain(X,N7), N7=<1.0.
# ab9(X) :- joiningyear(X,N1), N1=<2016.0, paymenttier(X,N3), N3=<2.0, not ab2(X), not ab3(X), not ab7(X), not ab8(X).
# ab10(X) :- education(X,'masters'), not everbenched(X,'no'), age(X,N4), N4=<27.0.
# ab11(X) :- age(X,N4), N4>26.0.
# ab12(X) :- gender(X,'female'), everbenched(X,'no'), paymenttier(X,N3), N3=<2.0, age(X,N4), N4>24.0, age(X,N4), N4=<26.0, experienceincurrentdomain(X,N7), N7=<4.0.
# ab13(X) :- not gender(X,'male'), joiningyear(X,N1), N1>2016.0, not ab11(X), not ab12(X).
# ab14(X) :- education(X,'masters'), city(X,'bangalore'), paymenttier(X,N3), N3>1.0, age(X,N4), N4=<27.0, not ab13(X).
# ab15(X) :- not city(X,'new_delhi'), city(X,'pune'), everbenched(X,'no'), joiningyear(X,N1), N1>2015.0, paymenttier(X,N3), N3=<2.0, experienceincurrentdomain(X,N7), N7=<4.0.
# ab16(X) :- education(X,'masters'), not city(X,'bangalore'), gender(X,'male'), joiningyear(X,N1), N1>2012.0, paymenttier(X,N3), N3>1.0, age(X,N4), N4>25.0, age(X,N4), N4=<26.0, not ab15(X).
# ab17(X) :- joiningyear(X,N1), N1=<2014.0, age(X,N4), N4>40.0.
# ab18(X) :- experienceincurrentdomain(X,N7), N7>5.0.
# ab19(X) :- not gender(X,'male'), city(X,'pune'), education(X,'bachelors'), not ab17(X), not ab18(X).
# ab20(X) :- education(X,'masters'), city(X,'bangalore'), paymenttier(X,N3), N3=<1.0, age(X,N4), N4=<26.0.
# ab21(X) :- gender(X,'female'), everbenched(X,'no'), paymenttier(X,N3), N3>2.0, paymenttier(X,N3), N3=<3.0, experienceincurrentdomain(X,N7), N7=<2.0.
# ab22(X) :- city(X,'new_delhi'), joiningyear(X,N1), N1>2016.0, age(X,N4), N4=<24.0, not ab21(X).
# ab23(X) :- city(X,'new_delhi'), gender(X,'female'), everbenched(X,'no'), joiningyear(X,N1), N1>2016.0, paymenttier(X,N3), N3>2.0, paymenttier(X,N3), N3=<3.0, age(X,N4), N4=<24.0, experienceincurrentdomain(X,N7), N7=<2.0.
# ab24(X) :- education(X,'masters'), not city(X,'bangalore'), not gender(X,'male'), joiningyear(X,N1), N1>2015.0, paymenttier(X,N3), N3>1.0, age(X,N4), N4>22.0, age(X,N4), N4=<25.0, not ab22(X), not ab23(X).
# ab25(X) :- joiningyear(X,N1), N1>2015.0, age(X,N4), N4>28.0, age(X,N4), N4=<31.0.
# ab26(X) :- experienceincurrentdomain(X,N7), N7=<3.0.
# ab27(X) :- joiningyear(X,N1), N1=<2012.0, not ab26(X).
# ab28(X) :- education(X,'masters'), city(X,'bangalore'), paymenttier(X,N3), N3>2.0, age(X,N4), N4>27.0, not ab25(X), not ab27(X).
# % acc 0.8539 p 0.8438 r 0.9618 f1 0.899
# % foldr++ costs:  0:00:00.640120
