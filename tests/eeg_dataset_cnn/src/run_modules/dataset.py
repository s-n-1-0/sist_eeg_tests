import json
import numpy as np
class EEGDataset():
    def read_dataset(self,filename="prep.npz"):
        with open("tests/eeg_dataset_cnn/src/settings.json","r") as json_file:
            settings = json.load(json_file)
            work_path =  settings["work_path"]
            npz_path = f"{work_path}/dest/{filename}"
            prep_dataset = np.load(npz_path)
            self.train_x = prep_dataset["train_x"]
            self.train_y = prep_dataset["train_y"]
            self.valid_x = prep_dataset["valid_x"]
            self.valid_y = prep_dataset["valid_y"]
            self.test_x = prep_dataset["test_x"]
            self.test_y = prep_dataset["test_y"]
    
    def _eeg_random_generater(self,x:np.ndarray,y:np.ndarray,signal_size:int,batch_size:int):
        steps = (x.shape[0] // signal_size) // batch_size
        def get_range():
            end = np.random.randint(0,(steps - 1) * signal_size) + signal_size
            start = end - signal_size
            while(np.all(y[end,:] == 0)):
                end = np.random.randint(0,(steps - 1) * signal_size) + signal_size
                start = end - signal_size
            return (start,end)
        count = 0
        while count < steps:
            count += 1
            ranges = [get_range() for _ in range(batch_size)]
            yield np.array([x[start:end,:] for start,end in ranges]),np.array([y[end,:] for _ ,end in ranges])

    def make_train_generator(self,signal_size:int,batch_size:int):
        return self._eeg_random_generater(self.train_x,self.train_y,signal_size=signal_size,batch_size=batch_size)
    def make_valid_generator(self,signal_size:int,batch_size:int):
        return self._eeg_random_generater(self.valid_x,self.valid_y,signal_size=signal_size,batch_size=batch_size)
    def make_test_generator(self,signal_size:int,batch_size:int):
        return self._eeg_random_generater(self.test_x,self.test_y,signal_size,batch_size)