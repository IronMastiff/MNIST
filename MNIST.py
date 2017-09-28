from Util import  fit_data, model, predict, print_iamge

train_X, train_Y, test_X, test_Y = fit_data()
parameters = model( train_X, train_Y, test_X, test_Y )
predict( parameters, test_X, 6 )
print_iamge( test_X, test_Y, 6 )





