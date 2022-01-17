# PubTables-1M


## Quick Start: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CePiqlZfJa_tLTCzbatOKahWlyEAMgQa?usp=sharing)


Pre-trained weights: [Link](https://drive.google.com/drive/folders/1Ko4Trk48u99AAPNU41RcUKAoMP0BoDmU?usp=sharing)

## For Docker users

```bash
docker pull phamquiluan/table-transformer:latest
# or
docker build -t phamquiluan/table-transformer -f Dockerfile .

# train TSR
docker run -it --shm-size 8G --gpus all \
  -v <data-path>:/code/data \
  -v phamquiluan/table-transformer \
  python3 main.py --data_root_dir /code/data --data_type structure
```


## Code Installation
Create a virtual environment and activate it as follows
```
python -m venv env; source env/bin/activate
pip install -U pip

pip install -r requirements.txt
```


[Original README](https://github.com/microsoft/table-transformer)
