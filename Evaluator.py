import torch
from tqdm import tqdm
#from torcheval.metrics import MulticlassConfusionMatrix

class Evaluator:
	def __init__(self,
				criterion,
				test_data,
				num_classes=None,
				device='cpu',
				verbose=True):
		self.criterion = criterion
		self.test_data = test_data
		self.num_classes = num_classes
		self.device = device
		self.verbose = verbose

	def _test_epoch(self, model):
		model.eval()
		metric = []
		metric2 = []
		losses = []
		for X, Y in (pbar := tqdm(self.test_data, desc="loss=")):
			with torch.no_grad():
				X, Y = X.to(self.device), Y.to(self.device)
				y = model(X, Y)
				loss = self.criterion(y, Y)
				losses.append(loss.item())
				metric.append(y.cpu())
				metric2.append(Y[:,:,53].cpu())
				pbar.set_description(f"loss={losses[-1]}")
		return sum(losses) / len(losses), (torch.cat(metric), torch.cat(metric2))

	def test(self, model):
		if self.verbose:
			print("Testing...")
		return self._test_epoch(model)
	
	