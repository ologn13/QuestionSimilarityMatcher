# QuestionSimilarityMatcher
ML model to check if two questions have the same intent or not.

## How to run?
Note: Running via flask server is having OOM issues. So, prefer the below method.
0. cd QuestionSimilarityMatcher
1. pip install -r requirements.txt
2. Download model.hdf5 file in current directory (i.e. QuestionSimilarityMatcher)
3. Run python shell and import TrainedModel class from trained_model
4. TrainedModel.predict("question1 string", "question2 string") => will give yes/no depending upon whether the questions had same intent or not.
