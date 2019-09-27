# Run AllenNLP on Windows

Update(27 Sept. 2019): This repository provides starting code for running allenlnp on Windows. An example included in the repository is priority classification for crisis-related tweets (Here just includes a fixture of dataset for practice purpose).

### Requirments
- I use [conda](https://docs.anaconda.com/anaconda/install/windows/) with the code so make sure conda is installed in your system.
- The [official AllenNLP repository](https://github.com/allenai/allennlp) is cloned to your local file system.
- To better hacking the code, it is recommended to have a look all [excellent tutorials](https://github.com/allenai/allennlp/tree/master/tutorials) on how to start AllenNLP.

### Steps of Usage
1. Clone this repository and put the allennlp package (remember it is actually the one named allennlp  under the official repository you cloned) under the root of the repository.
2. Open "conda prompt" and install necessary packages as required by allennlp by typing the command: `pip install -r requirements.txt`
3. Type `python run_test.py` to test if allennlp is run successfully for the code example.
4. If it works, there should a tmp folder be generated, under which a folder named my_debug containing all details of training the priority classification model with allennlp.

### Important Notes
- In experiments/priority_crisis_classifier.json, error occurs when comments are included. Hence, you should avoid commenting in the file, although this is not a issue on Mac OS or Linux system.
- In allennlp_mylib/dataset_readers/priority_crisis_tweets.py, exception raises when using WordTokenizer. Temporarily, I replaced it with simple str.split function. This needs hacking for better solution. 