# Apply-LINDA-BN-on-predicting-next-event


- [x] Create pytorch dataset
- [x] Create pytorch data_loader
- [x] Create pytorch lstm
- [x] Create trainer (TrainingController)
- [x] Add validation
- [x] Do testset performance test
- [x] Add Lr scheduler 
- [x] Add training and testing accuracy plot.
- [x] Save model, traing parameters, testing data after training is done.
- [x] Load pre-trained model
- [x] Save and load pre-processed data.
- [x] Add sample prediction in model.
- [x] Predict by input data.
- [x] The prediction should be able to use dataset and dataloader
- [x] Repair the argmax will cause infiite <PAD> 
- [x] Create the requirement.text
- [ ] Read papers
- [ ] Read LINDA-BN implementation
- [ ] Apply LINDA-BN


[[Predicting Next]](https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html)

#### Training Screenshot
![](https://github.com/ChihchengHsieh/Apply-LINDA-BN-on-predicting-next-event/blob/master/TrainingScreenshot/NotebookScreenshot.png?raw=true)


#### For prediction

Prediction should accept a json file with 2D array. (seq of traces)

Then, load the model (Need to create another controller for accesss prediction?)

Prediction => Concat the output (Sample or argmax) for the next prediction, then output the final trace.



#### Installing driver for graphics:


##### To show the availible drivers:
```
ubuntu-drivers devices
```

##### Install the recommended:
```
sudo ubuntu-drivers autoinstall
```

##### Install specific driver:
```
sudo apt install nvidia-340
```


## M1 Bugs

#### Installing pyAgrum [Solved]

When I clone the sorce code from [[GitLab]](https://gitlab.com/agrumery/aGrUM/-/tree/master/). It doesn't detect mine Python path, so I had to generate the wheel b:

```
act wheel release pyAgrum -d $(pwd) --python3lib=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") --python3include=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") --python=3
```

after building the wheel, we have to install the wheen through pip command. However, it gave me `is not a supported wheel on this platform.` Exception.

The file name of the weel is `pyAgrum-0.20.1.9-cp39-cp39-macosx_11_2_arm64.whl`

and the print thorugh `distutils.util.get_platform()` is `macosx-11.2-arm64`, which match exactly the wheel needed.

I don't know why this exception is throwed. My only chance is to change the file name of wheel to `pyAgrum-0.20.1.9-cp39-cp39-macosx_11_0_arm64.whl`.

And it works.


### Installing PyTables (tables) 

`pip3 install tables` doesn't work on this case, so I clone it from [[Source]](https://github.com/PyTables/PyTables):
```
git clone https://github.com/PyTables/PyTables
cd PyTables
```
And install the deps through brew, 
```
brew install hdf5 c-blosc lzo bzip2
```

Checking the path for install packages:
```
brew info -q hdf5 c-blosc lzo bzip2|grep '/opt/homebrew'
```

Then, build the setup.py and install it.
```
python setup.py build --hdf5=/opt/homebrew/Cellar/hdf5/1.12.0_3 --use-pkgconfig=FALSE --blosc=/opt/homebrew/Cellar/c-blosc/1.21.0 --lzo=/opt/homebrew/Cellar/lzo/2.10 --bzip2=/opt/homebrew/Cellar/bzip2/1.0.8
python setup.py install --hdf5=/opt/homebrew/Cellar/hdf5/1.12.0_3

```

### Installing cvxopt

I follow the instructions from [[Doc]](https://cvxopt.org/install/#macos) (Without Homebrew, since I cannot find the path of `CVXOPT_SUITESPARSE_SRC_DIR` for homebrew version)

Commnads from instruction:
```
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
pushd SuiteSparse
git checkout v5.6.0
popd
export CVXOPT_SUITESPARSE_SRC_DIR=$(pwd)/SuiteSparse
git clone https://github.com/cvxopt/cvxopt.git
cd cvxopt
git checkout `git describe --abbrev=0 --tags`
```

Before we install the package, we have to modify something in the setup.py since we use glpk (Optinal Lib for cvxopt).

Using `brew info glpk` to check the path, then in setup.py:
```
# Set to 1 if you are installing the glpk module.
BUILD_GLPK = 1

# Directory containing libglpk (used only when BUILD_GLPK = 1).
GLPK_LIB_DIR = '/opt/homebrew/Cellar/glpk/5.0/lib'

# Directory containing glpk.h (used only when BUILD_GLPK = 1).
GLPK_INC_DIR = '/opt/homebrew/Cellar/glpk/5.0/include'

```

or just export as env var:
```
export BUILD_GLPK = 1
export GLPK_LIB_DIR = '/opt/homebrew/Cellar/glpk/5.0/lib'
export GLPK_INC_DIR = '/opt/homebrew/Cellar/glpk/5.0/include'
```

Finally, we can:
```
python setup.py install
```


