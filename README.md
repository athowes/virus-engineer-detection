# virus-engineer-detection

## Getting Started

First, install the requirements into your python environment:
```
pip install -r requirements.txt
```

Next, run data preprocessing:
```
python3 utils/virus_data_processor.py
```

Finally, you can run train and test:
```
python3 train_and_test.py
```


Testsets (too large for repo):
- engineered and natural plasmids can be downloaded from here: https://drive.google.com/file/d/1-2Yu9RZ8f5r0UDGzyI4nmfO6XK_JRQ3C/view?usp=sharing
- viral vectors from addgene: https://drive.google.com/file/d/1hLFHeGGbbvex5JkZMM6Lt7yE_9YCXrdx/view?usp=sharing
