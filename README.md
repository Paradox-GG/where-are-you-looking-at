# Introduction
This is a project for machine learning **class desgin**(in another word, **homework**).
Input an image of a human head, classifiers could predict where the human head is looking at(left, right, up and straight).<br>
<br>For example, this man is looking at the right side:<br>
![figure_look_right](https://github.com/Paradox-GG/where-are-you-looking-at/blob/main/figure_look_right.png)
<br>and this man is looking up:<br>
![figure_look_up](https://github.com/Paradox-GG/where-are-you-looking-at/blob/main/figure_look_up.png)

There are three classifiers for this task:<br>
* **naive design based on cnn (acc: 96.8%);**<br>
* **naive design based on rnn (acc: 78.4%);**<br>
* **fine tune hrnet18 (acc: 93.6%).**<br>

# Dataset
Dataset downloaded from: https://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/faces_4.tar.gz,<br>
and processed with dataset/data_preprocessing.py

# Train
    python train_net.py

# Evaluation
    python eval_net.py

# Acknowledgement
The parts which are related to "hrnet" heavly rely on https://github.com/HRNet/HRNet-Image-Classification
