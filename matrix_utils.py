# https://nlp.stanford.edu/pubs/glove.pdf
import numpy as np
import collections
import os

class matrix_utils:
	def __init__(self):
		pass


	def set_large_voca_matrix(self, data_path, top_voca=50000, window_size=5, sub_voca_len=5000, savepath=None):
		'''
		40만 x 40만 matrix 안만들고도 할 방법

		sub_voca_len x 40만 만들어서 0이 아닌것 x, y, value 만들고 
		40만 / sub_voca_len 번 돌리면 될듯.
		=> 완전한 테이블 만들지 않고도 학습셋은 만들 수 있음.
		'''
		if os.path.exists(savepath+'word2idx.npy') and os.path.exists(savepath+'idx2word.npy'):
			word2idx = self.load_data(savepath+'word2idx.npy', data_structure ='dictionary')
			idx2word = self.load_data(savepath+'idx2word.npy', data_structure ='dictionary')
		else:
			word2idx, idx2word = self.get_vocabulary(data_path, top_voca, savepath)


		with open(data_path, 'r') as f:
			word = (f.readline().split())	#text8은 하나의 줄이며 단어마다 띄어쓰기로 구분.

		for_sub_word2idx = list(zip(word2idx.keys(), word2idx.values()))
		for_sub_word2idx = sorted(for_sub_word2idx, key=lambda column: column[1]) #value 기준으로 정렬.

		from tqdm import tqdm
		total_data_set = None

		#for i in tqdm(range( int(np.ceil(top_voca/sub_voca_len)) ), ncols=50, position=0):
		for i in range( int(np.ceil(top_voca/sub_voca_len)) ):
			print(i+1, '/', int(np.ceil(top_voca/sub_voca_len)) )
			sub_word2idx = for_sub_word2idx[sub_voca_len * i: sub_voca_len * (i + 1)] 
			sub_word2idx = dict(sub_word2idx)

			#sub_voca co-occurence matrix
			sub_matrix = np.zeros([sub_voca_len, top_voca], dtype=np.int32)

			for index in tqdm(range(len(word)), ncols=50):
				# get center word
				center_word = word[index]

				# calc center_word's row of sub_matrix
				if center_word in sub_word2idx:
					matrix_row = sub_word2idx[center_word] - sub_voca_len*i

					# get context word
					left_window_word = word[max(index-window_size, 0):index] # left context 데이터가 window size보다 적은 경우 처리.
					right_window_word = word[index+1:min(index+window_size, len(word)-1)+1] # right context 데이터가 window size모다 적은 경우 처리.
					context_word = left_window_word + right_window_word # [window_size*2]

					# calc sub_matrix value(co-occurence)
					for context in context_word:
						# calc context_word's column of sub_matrix
						if context in word2idx:
							matrix_column = word2idx[context]
							sub_matrix[matrix_row, matrix_column] += 1
			
			dataset = self.make_dataset_except_zerovalue_and_unk(sub_matrix, add_row=sub_voca_len*i)
			if total_data_set is None:
				total_data_set = dataset
			else:
				total_data_set = np.concatenate((total_data_set, dataset), axis=0) # [N, 3]
			print(i, total_data_set.shape)
		
		if savepath is not None:
			if not os.path.exists(savepath):
				print("create save directory")
				os.makedirs(savepath)
			self.save_data(savepath+'total_data_set.npy', total_data_set)
			print("total_data_set save", savepath+'total_data_set.npy', total_data_set.shape)

		return total_data_set




	def set_matrix(self, data_path, top_voca=50000, window_size=5, voca_loadpath=None, savepath=None):
		if voca_loadpath is None:
			word2idx, idx2word = self.get_vocabulary(data_path, top_voca, savepath)
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

				# calc matrix value(co-occurence)
				for context in context_word:
					# calc context_word's column of matrix
					if context in word2idx:
						matrix_column = word2idx[context]
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

		word2idx = {}
		idx2word = {}

		table = collections.Counter(word).most_common(top_voca) #빈도수 상위 x-1개 뽑음. 튜플형태로 정렬되어있음 [("단어", 빈도수),("단어",빈도수)] 
		for index, data in enumerate(table):
			word2idx[data[0]] = index
			idx2word[index] = data[0]

		if savepath is not None:
			if not os.path.exists(savepath):
				print("create save directory")
				os.makedirs(savepath)
			self.save_data(savepath+'word2idx.npy', word2idx)
			print("word2idx save", savepath+'word2idx.npy')
			self.save_data(savepath+'idx2word.npy', idx2word)
			print("idx2word save", savepath+'idx2word.npy')

		return word2idx, idx2word


	def make_dataset_except_zerovalue_and_unk(self, matrix, add_row=0):
		# except UNK

		not_zero_index = np.where(matrix != 0) # x_index_list, y_index_list
		
		# shape: [data_set_length, 3], value: [x_index, y_index, value]
		dataset = list(zip(not_zero_index[0]+add_row, not_zero_index[1], matrix[matrix != 0]))
		return np.array(dataset)



	def load_data(self, path, data_structure = None):
		if data_structure == 'dictionary': 
			data = np.load(path, encoding='bytes').item()
		else:
			data = np.load(path, encoding='bytes')
		return data


	def save_data(self, path, data):
		np.save(path, data)