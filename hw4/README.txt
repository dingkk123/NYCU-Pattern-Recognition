My environment is to use the google colab to run my code, and the following is the version that google colab use.

scikit-learn                     1.2.2
torch                            2.3.0+cu121
torchvision                      0.18.0+cu121
tqdm                             4.66.4
pandas                           2.0.3

Training part:
    Check the file path and run the train.ipynb, and it would create a folder name "model_weights", which contains the weight of each epoch.
Inference part:
    Check the file path of the weight and the submission file name, and run the inference.ipynb, and the submission.csv would be created at the same directory.


