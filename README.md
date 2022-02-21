# PubTables-1M

![](https://user-images.githubusercontent.com/24642166/150664500-c8a8359b-12b0-4ea7-be8b-12f6cc773fd4.png)

- Quick Start: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CePiqlZfJa_tLTCzbatOKahWlyEAMgQa?usp=sharing)

- Pre-trained weights: [Link](https://drive.google.com/drive/folders/1Ko4Trk48u99AAPNU41RcUKAoMP0BoDmU?usp=sharing)


## Update: 
- Jan 23 2022:
  - Release pre-trained weights (20 epochs)
- Jan 17 2022: 
  - Release pre-trained weights (11 epochs).
  - Add docker training
  - Add streamlit 


## For Docker users

```bash
docker pull phamquiluan/table-transformer:latest
# or
docker build -t phamquiluan/table-transformer -f Dockerfile .

# train TSR
docker run -it --shm-size 8G --gpus all \
  -v <data-path>:/code/data \
  phamquiluan/table-transformer \
  python3 main.py --data_root_dir /code/data --data_type structure
```


## Code Installation
Create a virtual environment and activate it as follows
```
python3.7 -m venv env; source env/bin/activate
pip install -U pip

pip install -r requirements.txt
```

## For streamlit users

```bash 
streamlit run app.py
```

[Original README](https://github.com/microsoft/table-transformer)
