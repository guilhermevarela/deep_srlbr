'''
Created on Jan 30, 2018
	@author: Varela
	
	Using tensorflow's Coordinator/Queue
	Using batching

	Debugging

'''
import numpy as np 
import tensorflow as tf 

from datasets.data_embed import embed_input_lazyload, embed_output_lazyload  

TARGET_PATH='datasets/inputs/00/'
tfrecords_filename= TARGET_PATH + 'devel.tfrecords'


def read_and_decode(filename_queue):
	'''
		Decodes a serialized .tfrecords containing sequences
		args
			filename_queue: [n_files] tensor containing file names which are added to queue


	'''
	reader= tf.TFRecordReader()
	_, serialized_example= reader.read(filename_queue)

	# a serialized sequence example contains:
	# *context_features.: which are hold constant along the whole sequence
	#   	ex.: sequence_length
	# *sequence_features.: features that change over sequence 
	context_features, sequence_features= tf.parse_single_sequence_example(
		serialized_example,
		context_features={
			'T': tf.FixedLenFeature([], tf.int64)			
		},
		sequence_features={
			'ID':tf.VarLenFeature(tf.int64),			
			'PRED':tf.VarLenFeature(tf.int64),			
			'LEMMA': tf.VarLenFeature(tf.int64),
			'M_R':tf.VarLenFeature(tf.int64),
			'targets':tf.VarLenFeature(tf.int64)		
		}
	)

	return context_features, sequence_features


# def process_example(length,  idx, idx_pred, idx_lemma,  mr, secret ,targets, embeddings, klass_ind):
def process(embeddings, klass_ind, context_features, sequence_features):
	context_inputs=[]
	sequence_inputs=[]
	sequence_target=[]

	context_keys=['T']
	for key in context_keys:
		val32= tf.cast(context_features[key], tf.int32)
		context_inputs.append( val32	 )

	#Read all inputs as tf.int64	
	sequence_keys=['ID', 'M_R', 'PRED', 'LEMMA', 'targets']
	for key in sequence_keys:
		dense_tensor= tf.sparse_tensor_to_dense(sequence_features[key])
		if key in ['PRED', 'LEMMA']:
			dense_tensor1= tf.nn.embedding_lookup(embeddings, dense_tensor)
			sequence_inputs.append(dense_tensor1)

		# Cast to tf.float32 in order to concatenate in a single array with embeddings
		if key in ['ID', 'M_R']:
			dense_tensor1=tf.expand_dims(tf.cast(dense_tensor,tf.float32), 2)
			sequence_inputs.append(dense_tensor1)

		if key in ['targets']:			
			dense_tensor1= tf.nn.embedding_lookup(klass_ind, dense_tensor)
			Y= tf.squeeze(dense_tensor1,1 )
			sequence_inputs.append(tf.cast(dense_tensor1,tf.float32)) # SANITY CHECK
	
	X= tf.squeeze( tf.concat(sequence_inputs, 2),1) 
	return X, Y, context_inputs[0]

# https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data
def input_fn(filenames, num_epochs, embeddings, klass_ind):
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

	context_features, sequence_features= read_and_decode(filename_queue)	

	inputs, targets, length= process(embeddings, klass_ind, context_features, sequence_features)	
	
	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * BATCH_SIZE

	# https://www.tensorflow.org/api_docs/python/tf/train/batch
	input_batch, target_batch, length_batch=tf.train.batch(
		[inputs, targets, length], 
		batch_size=batch_size, 
		capacity=capacity, 
		dynamic_pad=True		
	)
	return input_batch, target_batch, length_batch


def forward(X, Wo, bo, sequence_length):		
	'''
		Computes forward propagation thru basic lstm cell

		args:
			X: [batch_size, max_time, feature_size] tensor (sequences shorter than max_time are zero padded)

			Wo: [hidden_size[-1], klass_size] tensor prior to softmax layer, representing observable weights 

			bo: [klass_size] tensor prior to softmax layer of size 

			sequence_length:[batch_size] tensor (int) carrying the size of each sequence 

		returns:
			Yo: [batch_size, max_time, klass_size] tensor

	'''


	basic_cell= tf.nn.rnn_cell.MultiRNNCell(
		[tf.nn.rnn_cell.BasicLSTMCell(hdsz) 
				for hdsz in HIDDEN_SIZE ]
	)

	# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
	outputs, states= tf.nn.dynamic_rnn(
			cell=basic_cell,
			inputs=X, 			
			sequence_length=sequence_length,
			dtype=tf.float32,
			time_major=False
		)

	# Performs 3D tensor multiplication by stacking Wo batch_size times
	# broadcasts bias factor
	O_2d = tf.reshape(outputs,[-1,HIDDEN_SIZE[-1]])
	Y_2d = tf.matmul(O_2d, Wo) + bo 
	Y_hat= tf.reshape(Y_2d,[batch_size,-1,KLASS_SIZE])
	return Y_hat


if __name__== '__main__':	
	EMBEDDING_SIZE=50 
	KLASS_SIZE=22
	
	FEATURE_SIZE=2*EMBEDDING_SIZE+2+KLASS_SIZE
	lr=1e-6
	BATCH_SIZE=5
	N_EPOCHS=1
	HIDDEN_SIZE=[128, 64]
	DISPLAY_STEP=25

	word2idx,  np_embeddings= embed_input_lazyload()		
	klass2idx, np_klassind= embed_output_lazyload()		

	embeddings= tf.constant(np_embeddings.tolist(), shape=np_embeddings.shape, dtype=tf.float32, name= 'embeddings')
	klass_ind= tf.constant(np_klassind.tolist(),   shape=np_klassind.shape, dtype=tf.int32, name= 'klass')
	batch_size= tf.constant(BATCH_SIZE, dtype=tf.int32,  name='batch_size')
	hidden_size= tf.constant(HIDDEN_SIZE, dtype=tf.int32,  name='hidden_size')
	
	#define variables / placeholders
	Wo = tf.Variable(tf.random_normal([HIDDEN_SIZE[-1], KLASS_SIZE], name='Wo')) 
	bo = tf.Variable(tf.random_normal([KLASS_SIZE], name='bo')) 


	#output metrics
	xentropy= tf.placeholder(tf.float32, name='loss')	
	accuracy= tf.placeholder(tf.float32, name='accuracy')	
	logits=   tf.placeholder(tf.float32, shape=(BATCH_SIZE,None, KLASS_SIZE), name='logits')
	inputs=   tf.placeholder(tf.float32, shape=(BATCH_SIZE,None, FEATURE_SIZE), name='inputs')    
	X=   			tf.placeholder(tf.float32, shape=(BATCH_SIZE,None, FEATURE_SIZE), name='X')    
	#Defines piepeline 
	with tf.name_scope('pipeline'):
		inputs, targets, sequence_length = input_fn([tfrecords_filename], N_EPOCHS, embeddings, klass_ind)

	with tf.name_scope('predict'):
		predict_op= forward(X, Wo, bo, sequence_length)

	with tf.name_scope('xent'):
		cost_op= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict_op, labels=targets))

	with tf.name_scope('train'):
		optimizer_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_op)

	#Evaluation
	with tf.name_scope('accuracy'):
		success_count_op= tf.equal(tf.argmax(predict_op,1), tf.argmax(targets,1))
		accuracy_op = tf.reduce_mean(tf.cast(success_count_op, tf.float32))	

	#Logs 
	writer = tf.summary.FileWriter('logs/basic_lstm/00')			
	tf.summary.histogram('Wo', Wo)
	tf.summary.histogram('bo', bo)
	tf.summary.histogram('logits', logits)
	tf.summary.histogram('inputs', inputs)
	tf.summary.scalar('cross_entropy', xentropy)
	tf.summary.scalar('accuracy', accuracy)
	merged_summary = tf.summary.merge_all()

	#Initialization 
	#must happen after every variable has been defined
	init_op = tf.group( 
		tf.global_variables_initializer(),
		tf.local_variables_initializer()
	)
	with tf.Session() as session: 
		session.run(init_op) 
		coord= tf.train.Coordinator()
		threads= tf.train.start_queue_runners(coord=coord)
		step=0		
		total_loss=0.0
		total_acc=0.0		
		writer.add_graph(session.graph)
		try:
			while not coord.should_stop():				

				features, Y, length = session.run([inputs, targets, sequence_length])

				# import code; code.interact(local=dict(globals(), **locals()))		
				_, Yhat, count, loss, acc = session.run(
					[optimizer_op,predict_op, success_count_op,cost_op, accuracy_op],
					feed_dict={
						X: features,
						sequence_length: length,
						targets: Y							
					}
				)

				
				total_loss+=loss 
				total_acc+= acc

				if step % DISPLAY_STEP ==0:					
					print('Iter={:5d}'.format(step+1),'avg. acc {:.2f}%'.format(100*total_acc/DISPLAY_STEP), 'avg. cost {:.6f}'.format(total_loss/DISPLAY_STEP))										
					total_loss=0.0 
					total_acc=0.0

					s= session.run(merged_summary,
						feed_dict={accuracy: acc, xentropy: loss, logits:Yhat}
					)
					writer.add_summary(s, step)
				step+=1
				
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')

		finally:
			#When done, ask threads to stop
			coord.request_stop()
			
		coord.request_stop()
		coord.join(threads)

