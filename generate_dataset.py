import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt 
import os 
import random
import time
import threading
import logging
import concurrent.futures

# Create a logging object
logging.basicConfig(filename='dataset.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Set the logger for the main thread
logger.addHandler(logging.StreamHandler())

# Set the logger for the executor threads
executor_logger = logging.getLogger('concurrent.futures')
executor_logger.addHandler(logging.StreamHandler())

def create_config(grid_resolution, config_id, room_dim = np.random.rand(3)*5+1, rt60_tgt = 0.2, source_position = None, custom_dir=None):
        
        if source_position == None: 
            source_position = np.random.rand(3) * room_dim
        # Randomize room dimensions 
        # Randomize absorbtion 

        start_time = time.time()

        if custom_dir == None:
            config_dir = f"{grid_resolution}data/config_{config_id}"
            os.makedirs(config_dir, exist_ok=True)

        # Calculate the number of grid points along each dimension 
        num_grid_points_x = int(np.ceil(room_dim[0] / grid_resolution))  
        num_grid_points_y = int(np.ceil(room_dim[1] / grid_resolution))
        num_grid_points_z = int(np.ceil(room_dim[2] / grid_resolution))

        # Create grid points with a slight offset to avoid the edges
        grid_points_x = np.linspace(grid_resolution / 2, room_dim[0] - grid_resolution / 2, num_grid_points_x)
        grid_points_y = np.linspace(grid_resolution / 2, room_dim[1] - grid_resolution / 2, num_grid_points_y)
        grid_points_z = np.linspace(grid_resolution / 2, room_dim[2] - grid_resolution / 2, num_grid_points_z)

        # Create the meshgrid
        grid_x, grid_y, grid_z = np.meshgrid(grid_points_x, grid_points_y, grid_points_z)
        

        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

        logging.info(f"Config ID: {config_id} initilising with: \n   Room dimensions: {', '.join([f'{round(dim, 2)}m' for dim in room_dim])}, Absorption: {round(e_absorption, 2)},\n   Max order: {max_order}, Grid sizes: {round(grid_x.size, 2)}m x {round(grid_y.size, 2)}m x {round(grid_z.size, 2)}m")

        # Create the room
        logging.debug(f"Creating room...")
        room = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)

        # Create microphone array 
        mic_locations = np.c_[grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]  # Creates (N, 3) array, N is the number of grid points

        # Add microphones individually
        logging.debug("Adding microphones...")
        for mic_loc in mic_locations: 
            room.add_microphone(mic_loc)

        # Impulse signal 
        logging.debug(f"Creating impulse signal...")
        impulse_signal = np.zeros(200)  
        impulse_signal[0] = 1.0

        
        logging.debug(f"Source position: {source_position}")
        room.add_source(source_position, signal=impulse_signal)

        logging.debug("Computing RIRs...")
        # Simulate
        
        room.simulate()
        logging.debug("Done simulating, saving...")

        wavefield_dir = os.path.join(config_dir, "wavefield")
        os.makedirs(wavefield_dir, exist_ok=True)

        # Save recorded signals (modified)
        for mic_index in range(room.mic_array.signals.shape[0]):

            wavefield_filename = os.path.join(wavefield_dir, f"wavefield_mic{mic_index}.npy")
            np.save(wavefield_filename, room.mic_array.signals[mic_index, :])  # Save the entire signal for the mic 

        logging.debug("Saving other data...")

        # Save room dimensions
        room_dim_filename = os.path.join(config_dir, "room_dim.txt")
        np.savetxt(room_dim_filename, room_dim)  

        # Mic locations
        room_mic_filename = os.path.join(config_dir, "mic_locations.txt")
        np.savetxt(room_mic_filename, mic_locations)  

        room_abs_filename = os.path.join(config_dir, "rt60_tgt.txt")
        with open(room_abs_filename, 'w') as f:
            f.write(str(rt60_tgt))  

        room_source_filename = os.path.join(config_dir, "source.txt")
        np.savetxt(room_source_filename, source_position)

        end_time = time.time()
        time_taken = end_time - start_time
        logging.info(f"Config ID: {config_id} completed in : {time_taken} seconds")

        time_taken_filename = os.path.join(config_dir, "time_taken.txt")
        with open(time_taken_filename, 'w') as f:
            f.write(str(time_taken))

        return time_taken

def create_config_wrapper(args):
    return create_config(*args)

def generate_dataset(grid_resolution, num_configs, multithread, max_workers):

    logging.info(f"Grid resolution: {grid_resolution}")
    resolution_start_time = time.time()

    if multithread:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            config_ids = list(range(num_configs))
            args = [(grid_resolution, config_id) for config_id in config_ids]
            results = executor.map(create_config_wrapper, args)
            for result in results:
                pass  # This will raise any exceptions from the worker processes
    else:
        for config_id in range(num_configs):
            logging.info(f"Creating config {config_id} / {num_configs}")
            create_config(grid_resolution, config_id)

    resolution_end_time = time.time()
    logging.info(f"\n\nTime taken for all configs {grid_resolution}: {resolution_end_time - resolution_start_time} seconds")
    return True

if __name__ == '__main__':
    # Parameters 
    grid_resolution = 2.0
    num_configs = 250
    max_workers = 8

    # Generate dataset 
    while True:

        success = False
        if max_workers > 1:
            try:
                success = generate_dataset(grid_resolution, num_configs, True, max_workers)
            except MemoryError:
                logging.warning(f"Memory error encountered, reducing max workers...")
                max_workers = max_workers // 2
                logging.info(f"Reducing max workers to {max_workers}")

        else:
            success = generate_dataset(grid_resolution, num_configs, False, max_workers)
            logging.info(f"Processing sequentially...")

        if success:
            # exit()
            # Reduce grid resolution 
            logging.info(f"Reducing grid resolution to {grid_resolution/2}")
            grid_resolution = grid_resolution / 2
            max_workers = max_workers // 2 

