
# Fourier Neural Operator for High-Quality Reverb Generation

This project focuses on optimizing and dynamically generating high-quality reverb using Fourier Neural Operators (FNOs) for efficient performance on lower-end hardware.

https://docs.google.com/document/d/1b9CEbe8Fds-Hz4iyBhn1aJ4fVyc5H4zROjrzYYX7JlY/edit?usp=sharing

## Project Overview

This artifact has been developed using the agile Cowboy software methodology (Hollar, 2006) to adapt to challenges in this speculative area while maintaining core objectives. The program implements an FNO architecture capable of quickly synthesizing accurate room impulse responses (RIRs) based on unseen parameters, outperforming traditional hard-coded methods in speed.

## Requirements

- Python 3.11.7
- pip

It's recommended to use a virtual environment with Python 3.11.7, as this is the supported version for this program.

## Installation

1. Clone the repository:
   
git clone https://github.com/Volcanex/dissertation-v2.git
cd dissertation-v2

2. Create and activate a virtual environment:

python -m venv myvenv
myvenv\Scripts\activate.bat  # On Windows
source myvenv/bin/activate  # On Unix or MacOS

3. Install packages
 
pip install -r requirements.txt

VM Setup:

sudo git clone https://github.com/Volcanex/dissertation-v2.git
cd dissertation-v2
sudo -i
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
pip install torch torchvision
