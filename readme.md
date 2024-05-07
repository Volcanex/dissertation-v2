You'll need Python 3.11.7 with pip to use this software. 

*It's useful to make a venv here with python Python 3.11.7, that's the supported version for this program*

To build the python enviroment for this project: 
Run in the command line: 

python -m venv myvenv
myvenv\Scripts\activate.bat
pip install -r requirements.txt

*To build the dataset from scratch run generate_dataset.py*

It will overwrite the pre-generated files in here. Be careful to not use any half-generated datasets. Generally this means deleting whatever the smallest resolution directiory it begins to make. 

*Use data_reader.py*

Use data_reader.py to inspect the dataset.

*Run simple_rnn*

Run simple_rnn to build a RNN model. You can change the parameters inside the file. Also use 
this file to compare the models to the original data using option three.

*VM Stuff*

sudo git clone https://github.com/Volcanex/finaldaydis.git
cd finaldaydis
sudo -i
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt   
pip install torch torchvision