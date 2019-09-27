# Run AllenNLP on Windows

Update(27 Sept. 2019): After I got everything set up on my Windows Desktop, I found allennlp is currently not available for Windows. I checked online to find if there is any tutorial on this issue and turned out to be nothing there. Hence,
this repository provides starting code for running allenlnp on Windows. An example included in the repository is priority classification for crisis-related tweets (Here just includes a fixture of dataset for practice purpose). You can adapts it to any project you have with allennlp if you expect to run on Windows. 
### Requirments
- I use [conda](https://docs.anaconda.com/anaconda/install/windows/) with the code so make sure conda is installed in your system.
- The [official AllenNLP repository](https://github.com/allenai/allennlp) is cloned to your local file system.
- To better hacking the code, it is recommended to have a look all [excellent tutorials](https://github.com/allenai/allennlp/tree/master/tutorials) on how to start AllenNLP.

### Steps of Usage
1. Clone this repository and put the allennlp package (remember it is actually the one named allennlp  under the official repository you cloned) under the root of the repository.
2. Open "conda prompt" and install necessary packages as required by allennlp by typing the command: `pip install -r requirements.txt`
3. Type `python train_test.py` to train the code example model. If it works, there should a tmp folder be generated, under which a folder named my_debug containing all details of training the priority classification model with allennlp.
4. Type `python predict_test.py` to make predictions on the fixture of dataset. If it works, there should a file named predicted-test be in the predictions folder.


### Important Notes
- In experiments/priority_crisis_classifier.json, error occurs when comments are included. Hence, you should avoid commenting in the file, although this is not a issue on Mac OS or Linux system.
- In allennlp_mylib/dataset_readers/priority_crisis_tweets.py, exception raises when using WordTokenizer. Temporarily, I replaced it with simple str.split function. This needs hacking for better solution. 
- The code is by default run on Windows with GPU supported. If you want to experiment training on CPU, go to experiments/priority_crisis_classifier.json and change cuda_device from 0 to -1 (or change --cuda-device from 0 to -1 in predict_test.py if you want to make predictions on CPU).