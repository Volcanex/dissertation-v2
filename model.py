import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import librosa
import time
from neuralop.models import FNO, TFNO
import torch.nn as nn
import torch

try:
    from data_reader import visualize_room
    import sounddevice as sd
    def convolve_play(npy_file, audio_data, sample_rate):
        print("Convolving...")
        data = np.load(npy_file)
        convolved_audio = np.convolve(audio_data, data)
        convolved_audio /= np.max(np.abs(convolved_audio))
        print("Playing...")
        sd.play(convolved_audio, sample_rate)
        sd.wait()  

    def convolve_play_data(rir_data, audio_data, sample_rate):
        print("Convolving...")
        convolved_audio = np.convolve(audio_data, rir_data)
        convolved_audio /= np.max(np.abs(convolved_audio))
        print("Playing...")
        try:
            sd.play(convolved_audio, sample_rate)
            sd.wait()
        except:
            print("Error playing audio on your machine.")

except OSError:
    def convolve_play(npy_file, audio_data, sample_rate):
        print("Can't play audio")
    
    def convolve_play_data(rir_data, audio_data, sample_rate):
        print("Can't play audio")

print(torch.cuda.is_available())
print("Start.")

def load_all_config_data(resolution, return_time_taken=False):
    config_data = []
    for filename in os.listdir(f"{resolution}data"):
        if filename.startswith("config_"):
            config_dir = os.path.join(f"{resolution}data", filename)
            config_data.append(load_config_data(config_dir, return_time_taken))
    return config_data

def load_config_data(config_dir, return_time_taken=False):
    wavefield_dir = os.path.join(config_dir, "wavefield")
    room_dim_file = os.path.join(config_dir, "room_dim.txt")
    mic_locations_file = os.path.join(config_dir, "mic_locations.txt")
    source_file = os.path.join(config_dir, "source.txt")
    rt60_file = os.path.join(config_dir, "rt60_tgt.txt")

    room_dim = np.loadtxt(room_dim_file)
    mic_locations = np.loadtxt(mic_locations_file)
    source_location = np.loadtxt(source_file)
    with open(rt60_file, "r") as file:
        rt60 = float(file.read())

    files = [f for f in os.listdir(wavefield_dir) if f.endswith(".npy")]
    rir_data = []
    for filename in files:
        filepath = os.path.join(wavefield_dir, filename)
        data = np.load(filepath)
        match = re.search(r'wavefield_mic(\d+)\.npy', filename)
        if match:
            mic_index = int(match.group(1))
            if mic_index < len(mic_locations):
                mic_location = mic_locations[mic_index]
                rir_data.append((mic_location, data))
            else:
                print(f"Warning: Microphone index {mic_index} is out of bounds. Skipping.")
                # Floating point division issue? Maybe? 
        else:
            print(f"Warning: Invalid filename format: {filename}. Skipping.")

    if return_time_taken:
        time_taken_file = os.path.join(config_dir, "time_taken.txt")
        with open(time_taken_file, "r") as file:
            time_taken = float(file.read())
        return source_location, room_dim, rt60, rir_data, time_taken
    
    return source_location, room_dim, rt60, rir_data

class RIRDataset(torch.utils.data.Dataset):
    def __init__(self, config_data, resolution):
        self.resolution = resolution
        self.data = []
        max_rir_length = 0
        for source_location, room_dim, rt60, rir_data, _ in config_data:
            for mic_location, rir in rir_data:
                source_location = source_location.reshape(-1)[:3]
                room_dim = room_dim.reshape(-1)[:3]
                mic_location = mic_location.reshape(-1)[:3]
                input_data = np.concatenate((source_location, room_dim, [rt60], mic_location))
                self.data.append((input_data, rir))
                max_rir_length = max(max_rir_length, len(rir))

        self.rirs = np.zeros((len(self.data), max_rir_length))
        for i, (_, rir) in enumerate(self.data):
            self.rirs[i, :len(rir)] = rir

        self.rir_mean = np.mean(self.rirs)
        self.rir_std = np.std(self.rirs)
        self.rirs = (self.rirs - self.rir_mean) / self.rir_std

    def __getitem__(self, index):
        input_data, rir = self.data[index]
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor((self.rirs[index] - self.rir_mean) / self.rir_std, dtype=torch.float32)
        return input_tensor, target_tensor

    def __len__(self):
        return len(self.data)
    
class RIRModelBase(torch.nn.Module):
    def __init__(self, resolution):
        super(RIRModelBase, self).__init__()
        self.resolution = resolution

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement the forward method.")                                                                

class FNOModel(RIRModelBase):
    def __init__(self, modes, width, output_size, resolution, rank):
        super(FNOModel, self).__init__(resolution)
        self.fno = TFNO(
            n_modes=modes,
            hidden_channels=width,
            in_channels=10,
            out_channels=1,
            factorization='tucker',
            rank=rank
        )
        self.fc = nn.Linear(1, output_size)

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        output = self.fno(x)
        output = output.squeeze(2).squeeze(2)
        output = self.fc(output)
        return output

class RIRModel(RIRModelBase):
    def __init__(self, input_size, hidden_size, num_layers, output_size, resolution):
        super(RIRModel, self).__init__(resolution)
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x.unsqueeze(1))
        output = self.fc(output.squeeze(1))
        return output

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: "+str(device).upper())
    model = model.to(device)

    best_loss = float('inf')
    best_model_weights = None
    losses = []

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(train_loader)

        losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.0f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = model.state_dict()

    model.load_state_dict(best_model_weights)
    return losses

def plot_loss(losses, num_epochs):
    plt.plot(range(1, num_epochs+1), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs vs Loss')
    plt.show()

def plot_moving_average(losses, window_size, num_epochs):
    moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size, num_epochs+1), moving_avg)
    plt.xlabel('Epochs')
    plt.ylabel('Moving Average Loss')
    plt.title('Epochs vs Moving Average Loss')
    plt.show()

def compare_rirs(model, config_data, audio_data, sample_rate, resolution):
    for i, (source_location, room_dim, rt60, rir_data, time_taken) in enumerate(config_data):
        print(f"Config {i+1}:")
        print(f"  Room dimensions: {room_dim}")
        print(f"  Source location: {source_location}")
        print(f"  RT60: {rt60}")
        print(f"  Time taken: {time_taken:.4f} seconds")

        mic_position = rir_data[0][0]
        input_data = torch.tensor(np.concatenate((source_location, room_dim, [rt60], mic_position)), dtype=torch.float32)

        

        t1 = time.time()
        with torch.no_grad():
            model_output = model(input_data.unsqueeze(0)).squeeze(0).numpy()
        t2 = time.time()

        print(f"Time taken for model to evaluate: {t2-t1:.4f} seconds")

        generated_rir_dir = f"model_output_{resolution}data/model_config_{i}"
        os.makedirs(generated_rir_dir, exist_ok=True)
        generated_rir_file = os.path.join(generated_rir_dir, "model_wavefield_mic0.npy")
        np.save(generated_rir_file, model_output)

        original_rir_data = rir_data[0][1]

        print("Original RIR:")
        convolve_play_data(original_rir_data, audio_data, sample_rate)

        print("Generated RIR:")
        convolve_play(generated_rir_file, audio_data, sample_rate)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(original_rir_data)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Original RIR')

        plt.subplot(1, 2, 2)
        plt.plot(model_output)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Generated RIR')

        config_dir = f"{resolution}data/config_{i}"
        visualize_room(config_dir)

        plt.tight_layout()
        plt.show()

        print("=" * 40)

def load_model():
    model_path = input("Enter the model file: ")
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model_type = checkpoint['model_type']
        resolution = checkpoint['resolution']
        
        if model_type == "rnn":
            input_size = checkpoint['input_size']
            hidden_size = checkpoint['hidden_size']
            num_layers = checkpoint['num_layers']
            output_size = checkpoint['output_size']
            model = RIRModel(input_size, hidden_size, num_layers, output_size, resolution)
        elif model_type == "fno":
            modes = checkpoint.get('modes', (16, 16))  # Default value if 'modes' is missing
            width = checkpoint.get('width', 64)  # Default value if 'width' is missing
            rank = checkpoint.get('rank', 0.1)  # Default value if 'rank' is missing
            
            # Load the state dictionary
            state_dict = checkpoint['state_dict']
            
            # Get the output_size from the state dictionary
            output_size = state_dict['fc.weight'].shape[0]
            
            # Create the FNOModel instance with the correct output_size
            model = FNOModel(modes, width, output_size, resolution, rank)
            
            # Load the state dictionary into the model
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model file not found at {model_path}")
        return None
    
def generate_model(resolution, model_type, num_folds=5):
    config_data = load_all_config_data(resolution, return_time_taken=True)
    dataset = RIRDataset(config_data, resolution)

    fold_sizes = [len(dataset) // num_folds] * num_folds
    fold_sizes[-1] += len(dataset) % num_folds  # Adjust the last fold size

    kfold = torch.utils.data.random_split(dataset, fold_sizes)

    for fold, fold_dataset in enumerate(kfold):
        print(f"Fold {fold+1}/{num_folds}")
        
        train_indices = list(range(len(dataset)))
        test_indices = list(fold_dataset.indices)
        for i in test_indices:
            train_indices.remove(i)
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        if model_type == "rnn":
            input_size = 10
            hidden_size = 1024
            num_layers = 3
            output_size = dataset.rirs.shape[1]
            model = RIRModel(input_size, hidden_size, num_layers, output_size, resolution)
        elif model_type == "fno":
            modes = (16, 16)  # Example modes value
            width = 64  # Example width value
            output_size = dataset.rirs.shape[1]  # Get the expected output size from the dataset
            rank = 0.1
            model = FNOModel(modes, width, output_size, resolution, rank)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        num_epochs = 10
        losses = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)
        
    plot_loss(losses, num_epochs)
    if num_epochs > 10:
        plot_moving_average(losses, 10, num_epochs)

    model_name = input("Enter the model name: ") + f"_{model_type}"
    model_path = model_name + ".pth"
    
    checkpoint = {
        'model_type': model_type,
        'state_dict': model.state_dict(),
        'resolution': resolution,
        'modes': modes,
        'width': width,
        'output_size': output_size,
        'rank': rank
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved as {model_path}")

    return model

def menu(model):
    while True:
        print("\nMenu:")
        print("1. Load Model")
        print("2. Generate Model")
        print("3. Compare RIRs")
        print("4. Exit")

        if model is not None:
            print(f"\Model loaded: {model.__class__.__name__}")
        else:
            print("\nNo model loaded.")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            model = load_model()
        elif choice == "2":
            resolution = str(float(input("Enter the resolution (e.g., 2.0): ")))
            model_type = input("Enter the model type (rnn/fno): ")
            model = generate_model(resolution, model_type)
        elif choice == "3":
            if model is None:
                print("No model loaded. Please load or generate a model first.")
            else:
                resolution = model.resolution
                audio_path = "test.wav"
                audio_data, sample_rate = librosa.load(audio_path, sr=16000)
                sample_rate = 16000
                config_data = load_all_config_data(resolution, return_time_taken=True)
                compare_rirs(model, config_data, audio_data, sample_rate, resolution)

        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
            
def main():
    model = None
    menu(model)

if __name__ == "__main__":
    main()
