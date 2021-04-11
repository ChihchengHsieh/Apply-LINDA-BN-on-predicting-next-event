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
- [ ] Add sample prediction in model.
- [ ] Predict by input data.
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





