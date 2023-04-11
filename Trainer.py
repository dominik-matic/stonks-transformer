from tqdm import tqdm
import torch
import glob
import os

class Trainer:
	def __init__(self,
				model,
				criterion,
				optimizer,
				train_data,
				valid_data=None,
				scheduler=None,
				save_train_losses=True,
				save_valid_losses=True,
				device='cpu',
				epoch_save_interval=10,
				snapshot_path="snapshots/snapshot.ss",
				model_folder="models/",
				delete_previous_best_model=True,
				verbose=True):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.train_data = train_data
		self.valid_data = valid_data
		self.scheduler = scheduler
		self.save_train_losses = save_train_losses
		self.save_valid_losses = save_valid_losses
		self.device = device
		self.epoch_save_interval = epoch_save_interval
		self.snapshot_path = snapshot_path
		self.model_folder = model_folder
		self.delete_previous_best_model = delete_previous_best_model
		self.verbose = verbose
		self.current_epoch = 0
		self.train_losses = []
		self.valid_losses = []
		self.lowest_valid_loss = None
		self.previous_best_model_name = ""

		if os.path.exists(self.snapshot_path):
			self._load_snapshot(self.snapshot_path)
		
	

	def _train_epoch(self):
		self.model.train()
		losses = []
		for X, Y in (pbar_train := tqdm(self.train_data, desc="loss = ", leave=False)):
			self.optimizer.zero_grad()
			X, Y = X.to(self.device), Y.to(self.device)
			y = self.model(X, Y)
			loss = self.criterion(y, Y)
			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())
			pbar_train.set_description(f"train loss = {losses[-1]:.8f}")
		if self.scheduler is not None:
			self.scheduler.step()
		return sum(losses) / len(losses)
	
	def _valid_epoch(self, data):
		self.model.eval()
		losses = []
		for X, Y in (pbar_valid := tqdm(data, desc="loss = ", leave=False)):
			with torch.no_grad():
				X, Y = X.to(self.device), Y.to(self.device)
				y = self.model(X, Y)
				loss = self.criterion(y, Y)
				losses.append(loss.item())
				pbar_valid.set_description(f"valid loss = {losses[-1]:.8f}")
		return sum(losses) / len(losses)



	def _save_snapshot(self, epoch):
		if self.verbose:
			tqdm.write(f"Saving snapshot to {self.snapshot_path}...", end=" ")
		snapshot = {"state_dict": self.model.state_dict(),
					"current_epoch": epoch,
					"train_losses": self.train_losses,
					"valid_losses": self.valid_losses,
					"lowest_valid_loss": self.lowest_valid_loss,
					"previous_best_model_name": self.previous_best_model_name,
					"scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None}
		torch.save(snapshot, self.snapshot_path)
		if self.verbose:
			tqdm.write("DONE.")
	
	def _load_snapshot(self, load_path):
		if self.verbose:
			print(f"Loading snapshot from {load_path}...", end=" ")
		snapshot = torch.load(load_path, map_location=self.device)
		self.model.load_state_dict(snapshot["state_dict"])
		self.current_epoch = snapshot["current_epoch"]
		self.train_losses = snapshot["train_losses"]
		self.valid_losses = snapshot["valid_losses"]
		self.lowest_valid_loss = snapshot["lowest_valid_loss"]
		self.previous_best_model_name = snapshot["previous_best_model_name"]
		if self.scheduler is not None:
			self.scheduler.load_state_dict(snapshot["scheduler_state"])
		if self.verbose:
			print("DONE")
			
	
	def _save_model(self, epoch):
		if self.delete_previous_best_model and os.path.exists(self.previous_best_model_name):
			os.remove(self.previous_best_model_name)
		save_path = self.model_folder
		if hasattr(self.model, "name"):
			save_path += self.model.name + f"_e{epoch}.pt"
		else:
			save_path += f"model_e{epoch}.pt"
		if self.verbose:
			tqdm.write(f"Saving model to {save_path}...", end=" ")
		self.previous_best_model_name = save_path
		torch.save(self.model.state_dict(), save_path)
		if self.verbose:
			tqdm.write("DONE.")

	def train(self, n_epochs):
		# maybe this initial loss calculation is unnecessary?
		if self.current_epoch == 0:
			if self.verbose:
				print("Calculating initial losses on training and valid sets")
			l = self._valid_epoch(self.train_data)
			self.train_losses.append(l)
			l = self._valid_epoch(self.valid_data)
			self.valid_losses.append(l)
		for i in (pbar := tqdm(range(self.current_epoch, n_epochs), desc=f"E{self.current_epoch} TL={self.train_losses[-1]:.8f} VL={self.valid_losses[-1]:.8f}")):
			l = self._train_epoch()
			self.train_losses.append(l)
			l = self._valid_epoch(self.valid_data)
			self.valid_losses.append(l)
			if self.lowest_valid_loss is None or l < self.lowest_valid_loss:
				self.lowest_valid_loss = l
				self._save_model(i)
			if i % self.epoch_save_interval == 0:
				self._save_snapshot(i)
			pbar.set_description(f"E{i + 1} TL={self.train_losses[-1]:.8f} VL={self.valid_losses[-1]:.8f}")
		self._save_snapshot(n_epochs)

def main():
	pass

if __name__ == '__main__':
	main()