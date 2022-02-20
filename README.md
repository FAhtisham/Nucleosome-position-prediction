# Nucleosome-positoin-prediction


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

- GC Content,
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
