## Reproduce the results

### Using python3.9 environment

Setup environment and install all needed requirements

```sh
python3.9 -m venv .kvasir_seg_venv
source .kvasir_seg_venv/bin/activate
git clone https://github.com/GerasimovIV/kvasir-seg.git
cd kvasir-seg
pip install -r requirements.txt
```
Download and extarct datasets
```sh
make datasets
```
Download and extract checkpoint
```sh
make checkpoints
```
now you are ready to use any scripts or notebook 🚀
