# https://nlp.stanford.edu/pubs/glove.pdf
import tensorflow as tf
import collections

class GloVe:
	def __init__(self):
		self.matrix = None





	def set_matrix(self, data_path, top_voca=50000):
		#self.matrix = @@
		



	def get_vocabulary(self, data_path, top_voca=50000):
		with open(data_path, 'r') as f:
			word = (f.readline().split())	#text8은 하나의 줄이며 단어마다 띄어쓰기로 구분.

		word2idx = {'UNK':0}
		idx2word = {0:'UNK'}

		table = collections.Counter(word).most_common(top_voca-1) #빈도수 상위 x-1개 뽑음. 튜플형태로 정렬되어있음 [("단어", 빈도수),("단어",빈도수)] 
		for index, data in enumerate(table):
			word2idx[data[0]] = index+1
			idx2word[index+1] = data[0]

		return word2idx, idx2word
