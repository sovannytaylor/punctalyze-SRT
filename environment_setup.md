### commands for bioimage-fast build, run one line at a time in your terminal
```bash
conda create -y -n bioimage-fast -c conda-forge python=3.12 
conda activate bioimage-fast
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::ipykernel jupyter matplotlib
conda install conda-forge::numpy pandas scikit-image scipy loguru
python -m pip install cellpose --upgrade
conda install conda-forge::napari matplotlib-scalebar
conda install conda-forge::seaborn statannotations
pip install bioio bioio-ome-tiff bioio-ome-zarr bioio-czi bioio-nd2
conda install numpy=1.26 # later had to run this due to numpy issues
```