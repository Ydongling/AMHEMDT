# AMHEMDT
This is our Python implementation for the unpublished paper.
Prediction of miRNA-disease multi-type association based on attributed multi-layer heterogeneous network embedding

## Introduction
AMHEMDT is a model based on attributed multi-layer heterogeneous network embedding to learn the latent representations of miRNAs and diseases on each relation type, and then predict the association type of miRNA-disease by random forests.
## Environment Requirement
The code has been tested running under Python 3. The required packages are as follows:
   * TensorFlow >= 1.8 or PyTorch

## Arguments

	
* 'train_sample_neg_embedding','test_neg_classifier_index'
	+ the comfirmed miRNA-disease-type triple
	
* 'train_pos_classifier_index','train_sample_pos_embedding'
    + the unknown miRNA-disease-type triple
	
* 'feature_dic'
   + the feature of miRNA-disease paris