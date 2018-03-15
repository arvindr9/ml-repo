import numpy as np

X_adv_test = np.array(np.load('adv_test1.npz')['arr_0'])
x_adv_test = np.array(np.load('adv_test2.npz')['arr_0'])
X_adv_test = np.concatenate((X_adv_test, x_adv_test))

X_adv_train = np.array(np.load('adv_train1.npz')['arr_0'])
for i in range(2, 12):
    file = f'adv_train{i}.npz'
    X_test_adv = np.concatenate((X_adv_train, np.array(np.load(file)['arr_0'])))

print(X_adv_test.shape, X_adv_train.shape)

np.savez('adv_test.npz', X_adv_test)
np.savez('adv_train.npz', X_adv_train)
