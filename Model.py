import torch
from torch.nn import Linear
from torch.nn import Transformer

class Autoencoder_v1(torch.nn.Module):
	def __init__(self, num_features, layer_sizes):
		super(Autoencoder_v1, self).__init__()
		self.name = "Autoencoder_v1"
		
		self.encoder_layers = [Linear(num_features, layer_sizes[0])]
		for i in range(len(layer_sizes) - 1):
			self.encoder_layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
		self.encoder_layers.append(Linear(layer_sizes[-1], 1))

		self.decoder_layers = [Linear(1, layer_sizes[-1])]
		for i in reversed(range(len(layer_sizes) - 1)):
			self.decoder_layers.append(Linear(layer_sizes[i + 1], layer_sizes[i]))
		self.decoder_layers.append(Linear(layer_sizes[0], num_features))

		self.encoder_layers = torch.nn.ParameterList(self.encoder_layers)
		self.decoder_layers = torch.nn.ParameterList(self.decoder_layers)
		self.relu = torch.nn.ReLU()

	def encode_forward(self, X):
		y = X
		for layer in self.encoder_layers:
			y = layer(y)
			y = self.relu(y)
		return y
	
	def decode_forward(self, X):
		y = X
		for layer in self.decoder_layers[:-1]:
			y = layer(y)
			y = self.relu(y)
		return self.decoder_layers[-1](y)
	
	def forward(self, X):
		y = self.encode_forward(X)
		return self.decode_forward(y)

		



class Stonks(torch.nn.Module):
	def __init__(self,
				d_model=498,
				nhead=6,
				num_encoder_layers=6,
				num_decoder_layers=6,
				dim_feedforward=2048,
				dropout=0.1):
		super(Stonks, self).__init__()
		self.name = "Stonks_v1"
		self.transformer = Transformer(
							d_model=d_model,
							nhead=nhead,
							num_encoder_layers=num_encoder_layers,
							num_decoder_layers=num_decoder_layers,
							dim_feedforward=dim_feedforward,
							dropout=dropout,
							batch_first=True)
		self.fc = Linear(d_model, 1)
		

	def forward(self, src, tgt):
		#print(f'{src.shape=}, {tgt.shape=}')
		out = self.transformer(src, tgt)
		#print(f'{out.shape=}')
		out = self.fc(out)
		out = torch.squeeze(out, dim=2)
		#print(f'{out.shape=}')
		return out




def main():
	pass
	


if __name__ == '__main__':
	main()