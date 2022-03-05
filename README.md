# PubTables-1M

![](https://user-images.githubusercontent.com/24642166/150664500-c8a8359b-12b0-4ea7-be8b-12f6cc773fd4.png)

- Quick Start: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CePiqlZfJa_tLTCzbatOKahWlyEAMgQa?usp=sharing)

- Pre-trained weights: [Link](https://drive.google.com/drive/folders/1Ko4Trk48u99AAPNU41RcUKAoMP0BoDmU?usp=sharing)


## Update: 
- Mar 5 2022: 
  - Release pre-trained weights (12 epochs on full training set)
- Mar 1 2022: 
  - Release pre-trained weights (7 epochs on full training set)
  - Update core.py and google colab for new code
- Jan 23 2022:
  - Release pre-trained weights (20 epochs on a small set of data)
- Jan 17 2022: 
  - Release pre-trained weights (11 epochs on a small set of data).
  - Add docker training
  - Add streamlit 

## Evaluation of pre-trained epoches

### Epoch 12 - training on full training set

```
IoU metric: bbox                                                                                                                                 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.889                                                                                                                                                                                                 
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.963
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.931
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.671
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.882
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.892
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.847
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.925
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.729
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.917
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.930
pubmed: AP50: 0.963, AP75: 0.931, AP: 0.889, AR: 0.925
```


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
