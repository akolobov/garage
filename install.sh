conda create -n gym pip python=3.7 -y
conda activate gym

# Requirements
sudo apt-get update
sudo apt-get install -y gcc libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig libglew-dev

# Install gym
git clone https://github.com/openai/gym.git
cd gym
pip install -e .


# Install Mujoco
cd ~
wget https://www.roboti.us/download/mujoco200_linux.zip  --no-check-certificate
# sudo apt install unzip
mkdir ~/.mujoco
#mkdir ~/opt/.mujoco
#unzip mujoco200_linux.zip -d /opt/.mujoco
unzip mujoco200_linux.zip -d ~/.mujoco
rm mujoco200_linux.zip
cp -r ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
# mv /opt/.mujoco/mujoco200_linux /opt/mujoco200_linux
cp mjkey.txt ~/.mujoco
#cp mjkey.txt /opt
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin" >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mujoco200_linux/bin
#export MUJOCO_PY_MJKEY_PATH=/opt/mjkey.txt
#export MUJOCO_PY_MUJOCO_PATH=/opt/mujoco200_linux
pip install -U 'mujoco-py<2.1,>=2.0'
# python -c 'import mujoco_py'

# Install d4rl
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
pip install -U oyaml torch=1.7.1




git clone https://github.com/akolobov/garage.git
cd garage
git checkout conservative_ac
pip install -r requirements.txt
pip install -e '.[all,dev]'