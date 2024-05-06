import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import math
from itertools import combinations, product
import librosa
import glob
import os
import sounddevice as sd

# Redudant?
def load_source_position(config_dir):
    source_filename = os.path.join(config_dir, 'source.txt')
    try:
        source_position = np.loadtxt(source_filename)
        return source_position
    except Exception as e:
        print(f"Error loading source position: {e}")
        return None

# Function to visualize the wavefield (modified for 3D plot)
def visualize_wavefield_by_source_proximity(config_dir):

    wavefield_dir = os.path.join(config_dir, "wavefield")

    # Load source position
    source_position = load_source_position(config_dir)
    if source_position is None:
        return

    fig = plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    ax = fig.add_subplot(111, projection='3d')

    # Duplicate code?
    files = [f for f in os.listdir(wavefield_dir) if f.endswith(".npy")]
    for filename in files:
        filepath = os.path.join(wavefield_dir, filename)
        try:
            data = np.load(filepath)

            # Calculate microphone position (find its index in mic_locations.txt)
            mic_index = int(filename.split('wavefield_mic')[1].split('.npy')[0])
            mic_locations_filename = os.path.join(config_dir, "mic_locations.txt") 
            mic_positions = np.loadtxt(mic_locations_filename)
            mic_position = mic_positions[mic_index]


            distance = np.linalg.norm(mic_position - source_position) 

            # Create an array of distances with the same size as data
            distance_array = np.repeat(distance, len(data))

            time = np.arange(len(data))

            ax.scatter(time, data, distance_array, s=10, c=distance_array, cmap='viridis')

            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Amplitude")
            ax.set_zlabel("Proximity to source") 
            ax.set_title("Microphone Signals")

        except Exception as e:
            print(f"Error loading file {filename}: {e}")

    plt.show()


def visualize_room(config_dir):

    room_dim_file = os.path.join(config_dir, "room_dim.txt")
    mic_locations_file = os.path.join(config_dir, "mic_locations.txt")
    source_file = os.path.join(config_dir, "source.txt")
    
    # Load data from files
    room_dim = np.loadtxt(room_dim_file)
    mic_locations = np.loadtxt(mic_locations_file)
    source_location = np.loadtxt(source_file)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot wireframe of room
    for s, e in combinations(np.array(list(product([0, room_dim[0]], [0, room_dim[1]], [0, room_dim[2]]))), 2):
        if np.sum(np.abs(s-e)) == room_dim[0] or np.sum(np.abs(s-e)) == room_dim[1] or np.sum(np.abs(s-e)) == room_dim[2]:
            ax.plot3D(*zip(s, e), color="b")

    # Plot microphone locations
    ax.scatter(mic_locations[:, 0], mic_locations[:, 1], mic_locations[:, 2], color='r', label='Microphones')

    # Plot source location     
    ax.scatter(source_location[0], source_location[1], source_location[2], color='g', label='Sources')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    ax.set_xlim(0, np.max(room_dim))
    ax.set_ylim(0, np.max(room_dim))
    ax.set_zlim(0, np.max(room_dim))

    plt.show()


# Function to visualize the wavefield
def visualize_wavefield(config_dir):

    wavefield_dir = os.path.join(config_dir, "wavefield")

    files = [f for f in os.listdir(wavefield_dir) if f.endswith(".npy")]
    num_files = len(files)

    # Calculate the number of rows and columns for the grid of subplots
    num_cols = int(math.sqrt(num_files))
    num_rows = num_files // num_cols
    if num_files % num_cols != 0:
        num_rows += 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))  # Adjust the figure size as needed

    for i, filename in enumerate(files):
        filepath = os.path.join(wavefield_dir, filename)
        try:
            data = np.load(filepath)

            # Ensure data has at least a 2D shape for plotting
            if len(data.shape) == 1:
                data = data[:, np.newaxis]

            row = i // num_cols
            col = i % num_cols
            ax = axs[row, col] if axs.ndim > 1 else axs[i]
            ax.plot(data[:, 0])
            ax.set_title("Mic Signal")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")

        except Exception as e:
            print(f"Error loading file {filename}: {e}")

    plt.tight_layout()
    plt.show()

def hear_wavefield(config_dir, audio_file, mic_number):

    print("Loading audio file...")
    rirfilename = os.path.join("wavefield_mic" + mic_number + ".npy")
            
    wavefield_dir = os.path.join(config_dir, "wavefield")
    # Load audio file
    audio_data, sample_rate = librosa.load(audio_file)
    
    def convolve_play(npy_file, audio_data, sample_rate):
        print("Convolving...")
        data = np.load(npy_file)

        # Convolve the audio with the room impulse response
        convolved_audio = np.convolve(audio_data, data)

        # Normalize the output (optional, but often recommended)
        convolved_audio /= np.max(np.abs(convolved_audio))  

        # Play audio
        print("Playing...")
        sd.play(convolved_audio, sample_rate)
        sd.wait()  # Wait for playback to finish

    npy_file = os.path.join(wavefield_dir, rirfilename)
    convolve_play(npy_file, audio_data, sample_rate)

def menu(resolution, config_id):
    
    print("Menu:")
    print("1. Visualize Wavefield by Source Proximity")
    print("2. Visualize Room")
    print("3. Visualize Wavefield")
    print("4. Hear audio convolved with microphone room impulse response")
    print("5. Hear reference audio")
    print("10. Exit")
    choice = input("Enter your choice: ")
    
    config_dir = f"{float(resolution)}data/config_{config_id}"  

    if choice == "1":
        visualize_wavefield_by_source_proximity(config_dir)

    elif choice == "2":
        visualize_room(config_dir)

    elif choice == "3":
        visualize_wavefield(config_dir)

    elif choice == "4":
        audio_file = 'test.wav'
        mic_number = input("Enter mic number: ")
        hear_wavefield(config_dir, audio_file, mic_number)

    elif choice == "5":
        audio_file = 'test.wav'
        print("Playing audio...")
        audio_data, sample_rate = librosa.load(audio_file)
        sd.play(audio_data, sample_rate)
        sd.wait()

    elif choice == "10":
        exit()

    else:
        print("Invalid choice. Please try again.")

        
if __name__ == "__main__":
    xkdc = False
    resolution = str(float(input("Enter grid resolution: ")))
    config_id = input("Enter config id: ")
    while True:
        if xkdc == True:
            with plt.xkcd():
                menu(resolution, config_id)
        else:
            menu(resolution, config_id)
