git clone -b qinwen/zero https://github.com/google/maxdiffusion.git
pip install -U --pre jax[tpu] --no-cache-dir  -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install libtpu-nightly==0.1.dev20241001   -f https://storage.googleapis.com/libtpu-releases/index.html
pip install git+https://github.com/google/jax
pip install requirements.txt
sudo chmod 777 /usr/local/lib/python3.10/dist-packages/
sudo chmod 777 /usr/local/bin/
