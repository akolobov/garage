conda create -n gym pip python=3.7 -y
conda activate gym


# gym
sudo apt-get install -y gcc libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
# mkdir repos
# cd ~/repos
git clone https://github.com/openai/gym.git
cd gym
pip install -e .


# Mujoco
cd ~
wget https://www.roboti.us/download/mujoco200_linux.zip  --no-check-certificate
# sudo apt install unzip
mkdir ~/.mujoco
unzip mujoco200_linux.zip -d ~/.mujoco
rm mujoco200_linux.zip
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
cp ~/mjkey.txt ~/.mujoco
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin" >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
pip install -U 'mujoco-py<2.1,>=2.0'
# python -c 'import mujoco_py'

# garage
cd ~
git clone https://github.com/akolobov/garage.git
cd garage
git checkout shortrl
pip install -e '.[all,dev]'
