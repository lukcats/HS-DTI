# HS-DTI: Drug-target Interaction Prediction Based on Hierarchical Networks and Multi-order

- Source code for the paper "HS-DTI: Drug-target Interaction Prediction Based on Hierarchical Networks and Multi-order".
- The  HS-DTI model  thoroughly recognize the signifificance of protein multi-order sequence information and molecular functional groups for DTI prediction. A hierarchical network is utilized to capture the functional group information on the reconstructed molecular graph. Meanwhile, the 1st-order and 2nd-order sequence information of protein is modeled and encoded into the amino acid residue *R*1*, R*2*...R**N* , and several convolution layers further extract different levels of protein feature.



<img src="https://github.com/lukcats/HS-DTI/tree/main/visal/model.png" />

- The code was built based on [graghDTA](https://github.com/thinng/GraphDTA), [DeepDTA](https://github.com/hkmztrk/DeepDTA) and [Hierarchical Graph Representation Learning](https://github.com/murphyyhuang/gnn_hierarchical_pooling). Thanks a lot for their code sharing!

### Dataset

All data used in this paper are publicly available. data/davis/folds/test_fold_setting1.txt, train_fold_setting1.txt;  data/davis/Y, ligands_can.txt, proteins.txt
data/kiba/folds/test_fold_setting1.txt, train_fold_setting1.txt;  data/kiba/Y, ligands_can.txt, proteins.txt
These file were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data

### Usage

To run the training procedure,

1. Run [conda env create -f environment.yaml]( ) to set up the envirnoment.
2. Create data in pytorch format: [python]() [create_data.py]()
3. Run [python training.py 0 0]()  to train the model.  Where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively; the second argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively. 
4. Run [python training_validation.py 0 0](). Same arguments as in "3". Train a prediction model", a model is trained on training data and chosen when it gains the best MSE for testing data.

