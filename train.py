import GloVe
import matrix_utils
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

from sklearn.manifold import TSNE #pip install scipy, scikit-learn
import matplotlib.pyplot as plt #pip install matplotlib

data_path = './text8/text8'
savepath = './npy/'
tensorflow_saver_path = './saver/'
drawpath = './image/'

top_voca = 50000
sub_voca_len = 10000
window_size = 10
embedding_size = 300
x_max = 100
lr = 0.05

def draw_most_word_pyplot(model, idx2word, most, picture_name):
	most_word = [idx2word[i] for i in range(most)]
	
	most_i_word_embedding = sess.run(model.i_word_embedding_table[:most])
	most_k_word_embedding = sess.run(model.k_word_embedding_table[:most]) 
	most_word_embedding = most_i_word_embedding + most_k_word_embedding
	
	plt.figure(figsize=(18,18))
	tsne = TSNE(perplexity = 30, n_components = 2, init='pca')
	low_dim_embed = tsne.fit_transform(most_word_embedding)

	for i, label in enumerate(most_word):
		x, y = low_dim_embed[i]
		plt.scatter(x,y)
		plt.annotate(label, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')

	plt.savefig(picture_name)
	plt.close()


def weighting_function(data, x_max):
	# data: [N, 1]

	# if x < x_max
	weighting = data.copy()
	weighting[data<x_max] = (data[data<x_max]/x_max)**(3/4)
	# else
	weighting[data>=x_max] = 1.0

	return weighting


def train(model, dataset, x_max, lr):
	batch_size = 256
	loss = 0

	np.random.shuffle(dataset)

	for i in tqdm(range( int(np.ceil(len(dataset)/batch_size)) ), ncols=50):
		batch = dataset[batch_size * i: batch_size * (i + 1)] # [batch_size, 3]

		i_word_idx = batch[:, 0:1] # [batch_size, 1]
		k_word_idx = batch[:, 1:2] # [batch_size, 1] 
		target = batch[:, 2:].astype(np.float32) # [batch_size, 1] # will be applied log in model
		weighting = weighting_function(target, x_max)

		train_loss, _ = sess.run([model.cost, model.minimize],
					{
						model.i_word_idx:i_word_idx, 
						model.k_word_idx:k_word_idx, 
						model.target:target, 
						model.weighting:weighting,
						model.lr:lr 
					}
				)
		loss += train_loss
		
	return loss/len(dataset)



def run(model, dataset, x_max, lr, idx2word, restore=0):

	if not os.path.exists(tensorflow_saver_path):
		print("create save directory")
		os.makedirs(tensorflow_saver_path)

	if not os.path.exists(drawpath):
		print("create draw directory")
		os.makedirs(drawpath)

	for epoch in range(restore+1, 140+1):
		train_loss = train(model, dataset, x_max, lr)
		print("epoch:", epoch, 'train_loss:', train_loss, '\n')

		if (epoch) % 5 == 0:
			draw_most_word_pyplot(model, idx2word, most=500, picture_name=drawpath+str(epoch))
			model.saver.save(sess, tensorflow_saver_path+str(epoch)+".ckpt")
		


sess = tf.Session()

matrix_utils = matrix_utils.matrix_utils()
model = GloVe.GloVe(
			sess = sess, 
			voca_size = top_voca, 
			embedding_size = embedding_size
		)

'''
# 이미 계산한 결과가 있으면 불러옴.
if os.path.exists(savepath+'matrix.npy') and os.path.exists(savepath+'word2idx.npy') and os.path.exists(savepath+'idx2word.npy'):
	word2idx = matrix_utils.load_data(savepath+'word2idx.npy', data_structure ='dictionary')
	idx2word = matrix_utils.load_data(savepath+'idx2word.npy', data_structure ='dictionary')
	matrix = matrix_utils.load_data(savepath+'matrix.npy')

# 처음 계산하는 경우 
else:
	word2idx, idx2word = matrix_utils.get_vocabulary(data_path, top_voca=top_voca, savepath=savepath)
	matrix = matrix_utils.set_matrix(data_path, top_voca=top_voca, window_size=window_size, voca_loadpath=savepath, savepath=savepath)

#matrix = matrix.astype(np.float32) # [top_voca, top_voca]
#matrix /= np.sum(matrix, axis=1, keepdims=True) # [top_voca, top_voca]
# keepdims=True 해줘야 row별로 나눔.

dataset = matrix_utils.make_dataset_except_zerovalue_and_unk(matrix)
#print(dataset)
#print(dataset[:2], dataset.shape)
#print(matrix)

run(model, dataset, x_max=x_max, lr=lr)
'''

'''
test = matrix_utils.set_large_voca_matrix(
			data_path=data_path, 
			top_voca=50000, 
			window_size=10, 
			sub_voca_len=5000*2,  
			savepath=savepath
		)
'''
# 이미 계산한 결과가 있으면 불러옴.
if os.path.exists(savepath+'total_data_set.npy') and os.path.exists(savepath+'word2idx.npy') and os.path.exists(savepath+'idx2word.npy'):
	#word2idx = matrix_utils.load_data(savepath+'word2idx.npy', data_structure ='dictionary')
	idx2word = matrix_utils.load_data(savepath+'idx2word.npy', data_structure ='dictionary')
	dataset = matrix_utils.load_data(savepath+'total_data_set.npy')


print('\ntop_voca', top_voca)
print('window_size', window_size)
print('embedding_size', embedding_size)
print('x_max', x_max)
print('lr', lr)
print('dataset', dataset.shape)

run(model, dataset, x_max=x_max, idx2word=idx2word, lr=lr)
