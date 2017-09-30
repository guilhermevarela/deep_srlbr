'''
	Author: Guilherme Varela

	motivation: primary liblinear test 
'''

if __name__ == '__main__':

	from liblinearutil import *
	# Read data in LIBSVM format
	y, x = svm_read_problem('../heart_scale')
 	m = train(y[:200], x[:200], '-c 4')
 	p_label, p_acc, p_val = predict(y[200:], x[200:], m)