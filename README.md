# COCCIMORPH

This is a pure Python 3 implementation of [COCCIMORPH](http://www.coccidia.icb.usp.br/coccimorph/), a system for the automatic diagnosis of *Coccidia* using morphological characteristics. A test dataset for *Eimeria* species for domestic fowl can be found [here](http://www.coccidia.icb.usp.br/uploadoocyst/coccimorph/tutorials/SevenSpeciesFowl.zip); for rabbits, there is [this dataset](http://www.coccidia.icb.usp.br/uploadoocyst/coccimorph/tutorials/ElevenSpeciesRabbit.zip).

## Installation

```bash
$ conda create --name coccimorph biopython
$ source activate coccimorph
$ python -m ipykernel install --name coccimorph --user
$ pip install pandas opencv-python numpy
```

Before running the scripts:

```bash
$ cd <BASEPATH>
$ git clone https://github.com/ricardoy/coccimorph
$ cd coccimorph
$ export PYTHONPATH=.
```

## Running the classification

First, a good threshold value should be found (and optionally, the image scale); the `segment.py` script should be used for that:

```bash
$ python coccimorph/segment.py -i <IMAGE> -t <integer between 0 and 255> [-s <image scale in pixels/micrometer>] [-output-binary <FILENAME>] [-output-segmented <FILENAME>]
```

([This document](http://www.coccidia.icb.usp.br/uploadoocyst/coccimorph/tutorials/Tutorial-1-On-Line-Diagnosis.pdf) provides information about how to choose the threshold and the image scale)

After choosing the threshold value, run the `classifier.py` script:

```bash
$ python coccimorph/classifier.py  -input-file <IMAGE> -t <THRESHOLD> [--fowl | --rabit] [-s <SCALE>]
```

If no problems occurred, an output similar to the following one should be shown:

```bash
Mean of curvature: 1.122e-02
Standard deviation from curvature: 4.200e-03
Entropy of curvature: 3.431e+02
Largest diameter: 2.062e+02
Smallest diameter: 1.501e+02
Symmetry based on first principal component: 1.302e-02
Symmetry based on second principal component: 5.623e-02
Total number of pixels: 2.396e+04
Entropy of image content: 1.194e+01
Angular second moment from co-occurrence matrix: 2.663e-04
Contrast from co-occurrence matrix: 3.905e+02
Inverse difference moment from co-occurrence matrix: 1.124e-01
Entropy of co-occurence matrix: 3.766e+01

Probability classification:
E. acervulina: 99.5308
E. mitis: 0.4692

Similarity classification:
E. acervulina: 71.5685
E. necatrix: 28.1404
E. mitis: 7.0517

```