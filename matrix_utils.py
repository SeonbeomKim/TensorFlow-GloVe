# https://nlp.stanford.edu/pubs/glove.pdf
import numpy as np
import collections
import os

class matrix_utils:
	def __init__(self):
		pass




	def set_matrix(self, data_path, top_voca=50000, window_size=5, voca_loadpath=None, savepath=None):
		if voca_loadpath is None:
			word2idx, idx2word = self.get_vocabulary(data_path, top_voca)
		else:
			word2idx = self.load_data(voca_loadpath+'word2idx.npy', data_structure ='dictionary')
			idx2word = self.load_data(voca_loadpath+'idx2word.npy', data_structure ='dictionary')

		# co-occurence matrix
		matrix = np.zeros([top_voca, top_voca], dtype=np.int32)

		with open(data_path, 'r') as f:
			word = (f.readline().split())	#text8은 하나의 줄이며 단어마다 띄어쓰기로 구분.

		from tqdm import tqdm	
		for index in tqdm(range(len(word)), ncols=50):
			# get center word
			center_word = word[index]

			# get context word
			left_window_word = word[max(index-window_size, 0):index] # left context 데이터가 window size보다 적은 경우 처리.
			right_window_word = word[index+1:min(index+window_size, len(word)-1)+1] # right context 데이터가 window size모다 적은 경우 처리.
			context_word = left_window_word + right_window_word # [window_size*2]

			# calc center_word's row of matrix
			if center_word in word2idx:
				matrix_row = word2idx[center_word]
			else:
				matrix_row = word2idx['UNK']

			# calc matrix value(co-occurence)
			for context in context_word:
				# calc context_word's column of matrix
				if context in word2idx:
					matrix_column = word2idx[context]
				else:
					matrix_column = word2idx['UNK']

				matrix[matrix_row, matrix_column] += 1
		
		print('대칭성 체크(A == A.T)', np.array_equal(matrix, matrix.T))
		
		if savepath is not None:
			if not os.path.exists(savepath):
				print("create save directory")
				os.makedirs(savepath)
			self.save_data(savepath+'matrix.npy', matrix)
			print("matrix save", savepath+'matrix.npy')

		return matrix
		



	def get_vocabulary(self, data_path, top_voca=50000, savepath=None):
		with open(data_path, 'r') as f:
			word = (f.readline().split())	#text8은 하나의 줄이며 단어마다 띄어쓰기로 구분.

		word2idx = {'UNK':0}
		idx2word = {0:'UNK'}

		table = collections.Counter(word).most_common(top_voca-1) #빈도수 상위 x-1개 뽑음. 튜플형태로 정렬되어있음 [("단어", 빈도수),("단어",빈도수)] 
		for index, data in enumerate(table):
			word2idx[data[0]] = index+1
			idx2word[index+1] = data[0]

		if savepath is not None:
			if not os.path.exists(savepath):
				print("create save directory")
				os.makedirs(savepath)
			self.save_data(savepath+'word2idx.npy', word2idx)
			print("word2idx save", savepath+'word2idx.npy')
			self.save_data(savepath+'idx2word.npy', idx2word)
			print("idx2word save", savepath+'idx2word.npy')

		return word2idx, idx2word


	def make_dataset_except_zerovalue_and_unk(self, matrix):
		# except UNK
		matrix = matrix[1:, 1:] 

		not_zero_index = np.where(matrix != 0) # x_index_list, y_index_list
		
		# shape: [data_set_length, 3], value: [x_index, y_index, value]
		# unk인 첫 row와 column은 지웠으므로 인덱스는 1 더해줘야함..
		dataset = list(zip(not_zero_index[0]+1, not_zero_index[1]+1, matrix[matrix != 0]))
		return np.array(dataset)



	def load_data(self, path, data_structure = None):
		if data_structure == 'dictionary': 
			data = np.load(path, encoding='bytes').item()
		else:
			data = np.load(path, encoding='bytes')
		return data


	def save_data(self, path, data):
		np.save(path, data)