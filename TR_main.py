import torch
import os
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Stonks
from Dataset import NASDAQ_DS_TR
from tqdm import tqdm
from torch.utils.data import Subset

def main():
	DS_PATH = 'dataset/nasdaq_csv_processed/TR_ready.csv'
	print('Loading data...')
	dataset = NASDAQ_DS_TR(input_seq_len=90, output_seq_len=7, ds_path=DS_PATH)
	
	"""
		DROP A RANDOM COLUMN, 499 columns -> 498 columns,
		since num_heads in transformer must divide embed_dim,
		which is the number of columns
	"""
	print(f'Dropping random column: {dataset.data.columns[356]}')
	dataset.data.drop([dataset.data.columns[356]], axis=1, inplace=True)

	train_idx, valid_idx, test_idx = dataset.get_train_test_valid_idx(0.5, 0.3, 0.2, group_spikyness=2, seed=1337)
	train, valid, test = Subset(dataset, train_idx), Subset(dataset, valid_idx), Subset(dataset, test_idx)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f'Using {device=}')

	lr = 1e-4
	gamma = 0.99
	batch_size = 64
	n_epochs=250

	dl_train = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
	dl_valid = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=False)
	dl_test = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)	
	
	model = Stonks().to(device)
	#model.state_dict = torch.load('models/Stonks_v1_e228.pt')

	# determines which stock we're predicting
	predicting_stock_idx = 53

	print(f"Predicting stock: {dataset.data.columns[predicting_stock_idx]}")

	def loss_func(y, Y):
		_Y = Y[:,:,predicting_stock_idx] # random stock
		#print(f'{y.shape=}')
		#print(f'{_Y.shape=}')
		return torch.mean(torch.mean((y - _Y)**2, axis=1), axis=0)

	criterion = loss_func
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

	trainer = Trainer(model=model,
					criterion=criterion,
					optimizer=optimizer,
					train_data=dl_train,
					valid_data=dl_valid,
					device=device,
					epoch_save_interval=1,
					verbose=True)
	
	# should load model saved in ./models for evaluator
	evaluator = Evaluator(criterion=criterion,
						test_data=dl_test,
						device=device,
						verbose=True)
	
	trainer.train(n_epochs)
	loss, metric = evaluator.test(model)

	pred, true = metric[0], metric[1]

	total = 0
	correct = 0
	for _p, _t in zip(pred, true):
		for p, t in zip(_p, _t):
			if p * t > 0:
				correct += 1
			total += 1

	print(f'\"Accuracy\": {correct/total}')

	write_to_file = True
	if write_to_file:
		print('Writing to file...')
		with open("result.txt", "w") as file:
			file.write(f'Final {loss=}')
			file.write('Metric:')
			for i in range(len(test)):
				file.write(f'T: {true[i]}\n')
				file.write(f'P: {pred[i]}\n\n')

if __name__ == '__main__':
	main()