import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class NASDAQ_DS_AE(torch.utils.data.Dataset):
	def __init__(self, folder_path):
		dfs = []
		for filename in os.listdir(folder_path):
			cols = list(pd.read_csv(folder_path + filename, nrows=1))
			df = pd.read_csv(folder_path + filename, index_col=None, usecols=[c for c in cols if c != 'Date'], dtype=float)
			dfs.append(df)
		self.data = pd.concat(dfs, axis=0, ignore_index=True)		
		self.data = self.data.dropna()
		self._load_scaler()

	def __getitem__(self, index):
		return torch.tensor(self.data.iloc[index], dtype=torch.double), torch.tensor(self.data.iloc[index], dtype=torch.double)

	def __len__(self):
		return len(self.data)
	
	def _load_scaler(self):
		if os.path.exists("AE_scaler.bin"):
			self.scaler = torch.load("AE_scaler.bin")


	def scale_data(self, train_idx):
		self.scaler = StandardScaler()
		self.scaler.fit(self.data.iloc[train_idx])
		torch.save(self.scaler, 'AE_scaler.bin')
		self.data = pd.DataFrame(self.scaler.transform(self.data), columns=self.data.columns)
		self.data.to_csv('dataset/nasdaq_csv_processed/AE_scaled.csv', index=False)
	
	def rescale_data(self, data):
		return self.scaler.inverse_transform(data)



	def test_scaler(self, train_idx):
		self.scaler = StandardScaler()
		self.scaler.fit(self.data.iloc[train_idx])
		torch.save(self.scaler, 'AE_scaler.bin')
		scaled_data = pd.DataFrame(self.scaler.transform(self.data), columns=self.data.columns)
		rescaled_data = pd.DataFrame(self.scaler.inverse_transform(scaled_data), columns=self.data.columns)
		print(f'{self.data=}')
		print(f'{scaled_data=}')
		print(f'{rescaled_data=}')
	


class NASDAQ_DS_TR(torch.utils.data.Dataset):
	
	"""
		Use folder_path only when constructing the DS for the first time
		from a folder. The DS will be constructed and saved to ds_path
		from where it can be subsequently loaded.
		Loading is done automatically from ds_path when folder_path is set
		to None (default). 
	"""
	def __init__(self, input_seq_len, output_seq_len, ds_path='dataset/nasdaq_csv_processed/TR.csv', folder_path=None, scaler_path=None, verbose=True):
		self.input_seq_len = input_seq_len
		self.output_seq_len = output_seq_len
		
		self.verbose = verbose
		if folder_path is not None:
			self._log("Constructing data from folder...")
			self._construct_ds_from_folder(folder_path, ds_path)

		self.data = pd.read_csv(ds_path, dtype=float)
		self.scaler = self._load_scaler(scaler_path)
	

	"""
		group_spikyness affects how many time sections will valid
		and test subsets be split into, for group_spikyness=3, the ds
		could be split in the following way, where everything that's
		not VALID or TEST is TRAIN
		|-----|VALID|---|TEST|----|TEST|--------|VALID|-----|VALID|---|
		group_spikyness is only approximate to how many time groups will
		each subset occupy
	"""
	def get_train_test_valid_idx(self, train_c, test_c, valid_c, group_spikyness=3, seed=420):
		#np.random.seed(seed)
		train_idx = list(np.arange(self.data.shape[0] - self.input_seq_len - self.output_seq_len))
		valid_idx = []
		test_idx = []

		total_size = len(train_idx)

		valid_group_size = int(valid_c * total_size / group_spikyness)
		test_group_size = int(test_c * total_size / group_spikyness)

		for i in range(group_spikyness):
			rand_idx = np.random.randint(0, len(train_idx) - valid_group_size, 1)[0]
			new_valid_idx = train_idx[rand_idx:(rand_idx+valid_group_size)]
			valid_idx.extend(new_valid_idx)
			train_idx = [idx for idx in train_idx if idx < new_valid_idx[0] - self.input_seq_len or idx > new_valid_idx[-1]]
			#train_idx = train_idx[:rand_idx] + train_idx[(rand_idx+valid_group_size):]
			
			rand_idx = np.random.randint(0, len(train_idx) - test_group_size, 1)[0]
			new_test_idx = train_idx[rand_idx:(rand_idx+test_group_size)]
			test_idx.extend(new_test_idx)
			train_idx = [idx for idx in train_idx if idx < new_test_idx[0] - self.input_seq_len or idx > new_valid_idx[-1]]
			#train_idx = train_idx[:rand_idx] + train_idx[(rand_idx+test_group_size):]
			
		return train_idx, valid_idx, test_idx




	def _construct_ds_from_folder(self, folder_path, ds_path):
		self._log("Loading files...")
		dfs = []
		cols = []
		feature = "High" # experiment with other features?
		dtypes = {"Date":str, feature:float}
		for filename in os.listdir(folder_path):
			df = pd.read_csv(folder_path + filename, usecols=["Date", feature], dtype=dtypes)
			df.dropna()
			dfs.append(df)
			cols.append(filename[:-4]) # exclude '.csv'
		
		self._log("Calculating start and end dates...")
		start_date = None
		end_date = None
		date_format = "%d-%m-%Y"
		for df in dfs:
			s_d = datetime.strptime(df['Date'].iloc[0], date_format)
			e_d = datetime.strptime(df['Date'].iloc[-1], date_format)
			
			if start_date is None or s_d < start_date:
				start_date = s_d
			if end_date is None or e_d > end_date:
				end_date = e_d

		total_days = (end_date - start_date).days + 1

		data_matrix = np.empty((total_days, len(cols)), dtype=float)
		data_matrix.fill(np.nan)

		self._log("Populating data_matrix...")
		for j, df in enumerate(dfs):
			df["Date"] = df["Date"].apply(lambda d: (datetime.strptime(d, date_format) - start_date).days)
			for _, row in df.iterrows():
				i = int(row["Date"])
				value = row[feature] # feature = "High", but could be something else
				data_matrix[i][j] = value

		self._log("Saving to file...")
		raw_data = pd.DataFrame(data_matrix, columns=cols)
		raw_data.dropna(how='all', inplace=True)
		raw_data.to_csv(ds_path, index=False)
	
	def _construct_unbroken_chain_map(self):
		from tqdm import tqdm
		df_shape = self.data.shape
		chain_map = np.zeros(df_shape)
		chain_map[-1] += self.data.iloc[-1].notna().to_numpy()
		for i in tqdm(range(df_shape[0] - 2, -1, -1)):
			for j in range(df_shape[1]):
				if np.isnan(self.data.iloc[i,j]):
					chain_map[i,j] = 0
				else:
					chain_map[i,j] = chain_map[i + 1,j] + 1
		return chain_map


	"""
		@length_coef
			determines how many of the earliest
			days to remove from the dataset, this is done
			because the earliest days in the dataset have
			very little actual data, most of it is nan,
			and nan values are unacceptable

		@width_coef
			determines how many companies should
			be removed from the dataset, this will be done
			based on how long of an unbroken chain of data
			they have, when width_coef companies are removed
			from the dataset, the rest will be truncated
			from the front to fit the company with the smallest
			unbroken chain
	"""
	def truncate_data(self, length_coef, width_coef, save_loc=None):
		assert(length_coef < 1)
		assert(width_coef < 1)

		self.data.drop(labels=np.arange(self.data.shape[0] * length_coef), axis=0, inplace=True)
		cm = self._construct_unbroken_chain_map()
		torch.save(cm, "cm.tmp")
		lengths = np.sort(cm[0])
		smallest_length = lengths[int(width_coef * lengths.shape[0])]
		cols_to_drop = []
		for i in range(cm.shape[1]):
			if cm[0][i] < smallest_length:
				cols_to_drop.append(i)
		self.data.drop(labels=self.data.columns[cols_to_drop], axis=1, inplace=True)
		self.data.drop(labels=np.arange(int(smallest_length), self.data.shape[0]), axis=0, inplace=True)

		if save_loc is not None:
			self.data.to_csv(save_loc, index=False)

	def _load_scaler(self, scaler_path):
		if scaler_path is not None and os.path.exists(scaler_path):
			return torch.load(scaler_path)
		return None

	def scale_data(self, train_idx, scaled_ds_save_path='dataset/nasdaq_csv_processed/TR_scaled.csv', scaler_path='TR_scaler.bin'):
		self._log("Scaling data...")
		self.scaler = StandardScaler()
		self.scaler.fit(self.data.iloc[train_idx])
		
		
		self._log(f"Saving scaler to {scaler_path}...")
		torch.save(self.scaler, scaler_path)
		
		self._log(f"Saving scaled ds to {scaled_ds_save_path}...")
		self.data = pd.DataFrame(self.scaler.transform(self.data), columns=self.data.columns)
		self.data.to_csv(scaled_ds_save_path, index=False)

	def rescale_data(self, data):
		if self.scaler is None:
			raise Exception("Can't do it fam. Scaler isn't set")
		return self.scaler.inverse_transform(data)

	def get_returns(self, start_idx):
		for i in range(start_idx, start_idx + self.output_seq_len):
			a = 0
			b = 0
			try:
				a = self.data.iloc[i + 1]
			except:
				print(f"Failed on first step: {i + 1}")
			try:
				b = self.data.iloc[i]
			except:
				print(f"Failed on second step: {i}")
				
		return [self.data.iloc[i + 1] - self.data.iloc[i] for i in range(start_idx, start_idx + self.output_seq_len)]

	def __getitem__(self, index):
		src = torch.Tensor(self.data.iloc[index:index+self.input_seq_len].to_numpy())
		tgt = torch.Tensor(self.get_returns(index+self.input_seq_len - 1))
		return src, tgt
	def __len__(self):
		return len(self.data) - self.output_seq_len

	def _log(self, *args):
		if self.verbose:
			print(*args)
	

def test_NASDAQ_DS_AE():
	ds = NASDAQ_DS_AE('dataset/nasdaq_csv/')
	train, valid, test = torch.utils.data.random_split(ds, [len(ds)-2_000_000, 1_000_000, 1_000_000], generator=torch.Generator().manual_seed(420))
	ds.test_scaler(train.indices)

def test_NASDAQ_DS_TR():
	ds = NASDAQ_DS_TR(1, 1, ds_path='dataset/nasdaq_csv_processed/TR.csv')
	train, valid, test = torch.utils.data.random_split(ds, [len(ds)-3000, 1500, 1500], generator=torch.Generator().manual_seed(420))
	ds.scale_data(train.indices) # not valid scaling since I can't use random_split from torch to actually split the data, must use custom


def visualize_data():
	import matplotlib.pyplot as plt
	ds = NASDAQ_DS_TR(1, 1, ds_path='dataset/nasdaq_csv_processed/TR.csv')
	datamap = ds.data.isna().to_numpy()
	plt.imshow(datamap, aspect='auto')
	plt.show()
	cm = ds._construct_unbroken_chain_map()
	cm = np.sort(cm, axis=1)
	X = np.arange(cm.shape[1])
	Y = np.arange(cm.shape[0])
	X, Y = np.meshgrid(X, Y)
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	ax.plot_surface(X, Y, cm)
	plt.show()

def test_ds_split():
	ds = NASDAQ_DS_TR(90, 7, ds_path='dataset/nasdaq_csv_processed/TR_ready.csv')
	train, valid, test = ds.get_train_test_valid_idx(0.5, 0.3, 0.2, group_spikyness=2, seed=1343)
	print(len(train))
	print(len(valid))
	print(len(test))

	print(set(train).isdisjoint(set(valid)))
	print(set(train).isdisjoint(set(test)))
	print(set(valid).isdisjoint(set(test)))

	import matplotlib.pyplot as plt

	xs = np.arange(ds.data.shape[0])

	plt.plot(xs, np.isin(xs, train))
	plt.show()
	plt.plot(xs, np.isin(xs, valid))
	plt.show()
	plt.plot(xs, np.isin(xs, test))
	plt.show()
	
def test_get_item():
	from torch.utils.data import Subset
	ds = NASDAQ_DS_TR(90, 7, ds_path='dataset/nasdaq_csv_processed/TR_ready.csv')
	ds.data.drop([356])
	train_idx, valid_idx, test_idx = ds.get_train_test_valid_idx(0.5, 0.3, 0.2, group_spikyness=2, seed=1337)
	train, valid, test = Subset(ds, train_idx), Subset(ds, valid_idx), Subset(ds, test_idx)

	for x, d in enumerate([train, valid, test]):
		for i in range(len(d)):
			try:
				a = d[i]
			except Exception as e:
				print(x)
				print(i)
				print(e)

	print("Done!")
if __name__ == '__main__':
	#test_NASDAQ_DS_TR()
	#visualize_data()

	#ds = NASDAQ_DS_TR(1, 1, ds_path='dataset/nasdaq_csv_processed/TR.csv')
	#ds.truncate_data(0.5, 0.7, save_loc='dataset/nasdaq_csv_processed/TR_ready.csv')
	
	#test_ds_split()
	test_get_item()

	
	
	
	
