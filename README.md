# NucBoost (Nucleosome-Position-Prediction-Via-Boosting)


This repository contain the codes for Numcleome position prediction.

## Dataset

- G1

  - C. elegans
  - D. melanogaster
  - H. sapiens

- G2

  - HM-LC (Long chr)
  - DM-LC (Long chr)
  - YS-WG (Whole Genome)
  - HM-PM (Promoter)
  - DM-PM (Promoter)
  - YS-PM (Promoter)
  - HM-5U (5UTR)
  - DM-5U (5UTR)


## Feature generation

NucBoost extracts 9 different feature vectors from a DNA sequence.

- GC Content
- k Tuple
- k Gap 
- Pseudo K Nucleotied Composition 
- zCurve
- GC Content 
- atgcRatio 
- Mono Mono 
- Mono Di 
- Mono Tri 
- Di Mono 
- Di Di 
- Di Tri 
- Tri Mono 
- Tri Di


## Feature Reduction

- Feature reduciton was performed using two algorithms
- AdaBOOST with SAMME.R, Depth=1, NEstimators = 500

##  Training

Different ML classifiers were trained on the above mentioned features. Whereas, AdaBoost seems to outperform state of the art approches.



## Scipts
- Feature Generation and Reduction
  - `python ~/Nucleosome_Prediction/python/PyFeat/Codes/main.py --sequenceType=DNA --fullDataset=1 --optimumDataset=1 --fasta=/home/abbasi/Nucleosome_Prediction/python/PyFeat/Datasets/DNA/G2/H5U_seqs.fasta --label=/home/abbasi/Nucleosome_Prediction/python/PyFeat/Datasets/DNA/G2/H5U_labels.txt --kTuple=3 --kGap=5 --pseudoKNC=1 --zCurve=1 --gcContent=1 --cumulativeSkew=1 --atgcRatio=1 --monoMono=1 --monoDi=1 --monoTri=1 --diMono=1 --diDi=1 --diTri=1 --triMono=1 --triDi=1`

- Training Classifiers
  - `python ~/Nucleosome_Prediction/python/PyFeat/Codes/runClassifiers.py --nFCV=10 --dataset=./optimumDataset.csv --auROC=1 --boxPlot=1`


