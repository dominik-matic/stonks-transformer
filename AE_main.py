import torch
import os
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Autoencoder_v1
from Dataset import NASDAQ_DS_AE
from tqdm import tqdm

"""
	This pretty much a failed attempt of trying to compress
	a six dimensional vector into a scalar using an AE.
"""


def main():
	SCALED_DS_PATH = 'dataset/nasdaq_csv_processed/AE_scaled.csv'

	dataset = None
	if not os.path.exists(SCALED_DS_PATH):
		print('Scaling data...')
		dataset = NASDAQ_DS_AE('dataset/nasdaq_csv/')
		train, valid, test = torch.utils.data.random_split(dataset, [len(dataset)-2_000_000, 1_000_000, 1_000_000], generator=torch.Generator().manual_seed(420))
		dataset.scale_data(train.indices)	

	print('Loading data...')
	dataset = NASDAQ_DS_AE('dataset/nasdaq_csv_processed/')
	train, valid, test = torch.utils.data.random_split(dataset, [len(dataset)-2_000_000, 1_000_000, 1_000_000], generator=torch.Generator().manual_seed(420))
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f'Using {device=}')

	lr = 1e-3
	gamma = 0.9
	batch_size = 5_000
	n_epochs=100

	dl_train = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
	dl_valid = torch.utils.data.DataLoader(dataset=valid, batch_size=batch_size, shuffle=False)
	dl_test = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)	
	
	model = Autoencoder_v1(6, [512, 512, 512]).double().to(device)

	criterion = lambda y, Y: torch.mean(torch.mean((y - Y)**2, axis=1), axis=0)

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
	
	evaluator = Evaluator(criterion=criterion,
						test_data=dl_test,
						num_classes=2,
						device=device,
						verbose=True)
	
	trainer.train(n_epochs)
	loss, metric = evaluator.test(model)

	trues = dataset.data.iloc[test.indices]

	print('Writing to file...')
	with open("result_unscaled.txt", "w") as file:
		for i in range(len(test)):
			file.write(f'T: {trues[i]}\n')
			file.write(f'P: {metric[i]}\n\n')

	trues = dataset.rescale_data(trues)
	metric = dataset.rescale_data(metric)


	print('Writing to file...')
	with open("result.txt", "w") as file:
		file.write(f'Final {loss=}')
		file.write('Metric:')
		for i in range(len(test)):
			file.write(f'T: {trues[i]}\n')
			file.write(f'P: {metric[i]}\n\n')

if __name__ == '__main__':
	main()