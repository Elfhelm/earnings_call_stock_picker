import os
import re
import csv

from pandas import DataFrame
from yahoo_finance import Share

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.base import TransformerMixin

NEWLINE = '\n'
SKIP_FILES = {'.DS_Store'}

SOURCES = [
    'Transcripts_TXT'
]

init_dates1 = {
    'Q1': '04-01',
    'Q2': '07-01',
    'Q3': '10-01',
    'Q4': '01-01'
}
init_dates2 = {
    'Q1': '04-04',
    'Q2': '07-04',
    'Q3': '10-04',
    'Q4': '01-04'
}

# These final dates are 3 months after the initial dates above
fin_dates1 = {
    'Q1': '06-30',
    'Q2': '09-30',
    'Q3': '12-28',
    'Q4': '03-31'
}
fin_dates2 = {
    'Q1': '07-04',
    'Q2': '10-04',
    'Q3': '12-31',
    'Q4': '04-04'
}

### These are 1 month after the initial dates
##fin_dates1 = {
##    'Q1': '04-30',
##    'Q2': '07-31',
##    'Q3': '10-31',
##    'Q4': '01-31'
##}
##fin_dates2 = {
##    'Q1': '05-04',
##    'Q2': '08-04',
##    'Q3': '11-04',
##    'Q4': '02-04'
##}

### 2 weeks after initial dates
##fin_dates1 = {
##    'Q1': '04-15',
##    'Q2': '07-15',
##    'Q3': '10-15',
##    'Q4': '01-15'
##}
##fin_dates2 = {
##    'Q1': '04-18',
##    'Q2': '07-18',
##    'Q3': '10-18',
##    'Q4': '01-18'
##}

# 4 for quarters, 12 for months
MARKET_PERIODS_PER_YEAR = 4

DATA_FILE = 'data_3months.csv'

stored_data = {}
with open(DATA_FILE, 'rt') as stored_data_file:
    data_reader = csv.reader(stored_data_file, delimiter=',')
    for row in data_reader:
        companyName = row[0]
        stored_data[companyName] = row[1:3]
    stored_data_file.close()

QRE = re.compile('Q[1234]')
YRE = re.compile('20[01]\d')

ROSE = 1
FELL = 0
        
##stored_data_file = open(DATA_FILE, 'a')

# utility function for reading all files on a given path
def readFiles(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content

# check if a quarter number is valid
def qValid(quarter):
    try:
        qnum, year = quarter.split('_')
        if QRE.match(qnum) and YRE.match(year) and int(year) < 2016:
            return 1
        else:
            return 0
    except ValueError:
        return 0
    
# check whether the stock rose or fell in a given quarter, storing and reusing data when possible
def getPriceChange(companyName, quarter):
    call_id = companyName + "_" + quarter
    if call_id in stored_data:
        init_price = float(stored_data[call_id][0])
        fin_price = float(stored_data[call_id][1])
##        if fin_price > init_price:
##            return 1
##        else:
##            return 0
        return (fin_price - init_price) / init_price
        
    s = Share(companyName)
    qnum, year = quarter.split('_')
    dinit1 = init_dates1[qnum]
    dinit2 = init_dates2[qnum]
    dfin1 = fin_dates1[qnum]
    dfin2 = fin_dates2[qnum]
    
    if (qnum == 'Q4'):
        year = str(int(year) + 1)
    if (qnum == 'Q3' and year == '2016'):
        dfin1 = '12-11'
        dfin2 = '12-12'

    dinit1 = year + '-' + dinit1
    dinit2 = year + '-' + dinit2
    dfin1 = year + '-' + dfin1
    dfin2 = year + '-' + dfin2
    print(dinit1)

    init_price = float(s.get_historical(dinit1, dinit2)[-1]['Open'])
    print(init_price)
    fin_price = float(s.get_historical(dfin1, dfin2)[-1]['Open'])

    with open(DATA_FILE, 'a') as stored_data_file:
        stored_data_file.write(companyName)
        stored_data_file.write("_")
        stored_data_file.write(quarter)
        stored_data_file.write(",")
        stored_data_file.write(init_price)
        stored_data_file.write(",")
        stored_data_file.write(fin_price)
        stored_data_file.write("\n")

    call_id = companyName + "_" + quarter
    stored_data[call_id] = [init_price, fin_price]

##    if (fin_price > init_price):
##        return 1
##    else:
##        return 0
    return (fin_price - init_price) / init_price

def getFeatures(text):
    return text

# parse each call transcript into useful text
def parseCall(comp, call_text):
    output = ''
    quarter = ''
    call_ID = comp
    call_lines = call_text.split('\n')
    sentence = 0
    heading = call_lines[0].split()
    for h in heading:
        if h[0][0] == 'Q':
            call_ID += "_" + h
            quarter += h + '_'
        elif h[0][0] == '2' or  h[0][0] == '1':
            call_ID += "_" + h
            quarter += h
            break
        elif h[0][0] == '0':
            call_ID += "_20" + h
            quarter += "20" + h
            break

    line = 0
    seg = ''
    speaker = ''
    call_lines = call_lines[15:]

    for l in call_lines:            
        output = output + str(l) + " "

    return output, quarter

# get company name, quarter, and useful reformatted text from raw call transcript files
def parseFile(fileName, text):
    companyName = fileName.split('_')[-1].split('.')[0]

    with open(fileName, 'r') as trans_file:
        content = trans_file.read()
        calls = content.split('\n HD\n')[1:] # Skip the first which is the table of contents
        for c in calls:
            callText, quarter = parseCall(companyName, c)

            yield companyName, quarter, callText

# highest level function - formats data as a Pandas data frame
def buildDataFrame(path):
    rows = []
    index = []
    for fileName, text in readFiles(path):
        print('Loading', fileName)
        for companyName, quarter, callText in parseFile(fileName, text):
            if quarter == 'Q4_2016':
                continue
            if not qValid(quarter):
                continue
            
            rows.append({'features': getFeatures(callText), 'class': getPriceChange(companyName, quarter)})
            index.append(companyName + '_' + quarter)

    data_frame = DataFrame(rows, index=index)
    return data_frame

# class for transforming data from sparse to dense matrix - needed for gradient boosting
class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

data = DataFrame({'features': [], 'class': []})
for path in SOURCES:
    data = data.append(buildDataFrame(path))

# remove duplicate entries
data = data.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')

# binarize price changes so we can classify stocks into 'buy' or 'do not buy'
# it's interesting to experiment with the threshold - here I've set the program to look for returns of
#    at least 0.5%
# however, the results don't seem to be much different from those at 0 or 1%
binarizer = Binarizer(threshold=0.005)
temp = data['class'].values.reshape(1, -1)
data['class'] = binarizer.transform(temp)[0]

# shuffle data
permutation = np.random.permutation(data.index)
data = data.reindex(permutation)

# ML pipeline
# Vectorizer:
#    basic CountVectorizer: counts words in earnings call text
#    bigrams: counts bigrams (needs feature selection to avoid overfitting)
# Variance threshold: removes features that are the same in x% of samples
#    K-best feature selection reduces performance substantially
# Tfidf transformer: converts word counts into proportions
# Classifier: classifies stocks (results below are with basic CountVectorizer and no feature selection)
#    Naive Bayes: tends to do about 0.5-1% better than the whole market; results are more stable than other algorithms
#    SVM: does about 0.5-1% better than the whole market (often advises buying every stock in the portfolio)
#    Logistic regression: probably does a little better than the whole market
#    Random forest classifier: probably does a little better than the whole market
#    Decision tree classifier: does a little worse than the whole market
#    K neighbors classifier: does about 1-2% worse than the whole market for various values of K
#    Naive Bayes-based bagging classifier: usually does around 1-2% better than the market
#    Gradient boost: best algorithm I've found, slow but usually does around 2-3% better than the market
pipeline = Pipeline([
##    ('vectorizer', CountVectorizer()),
    ('vectorizer', CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)),
    ('feature_selector', VarianceThreshold(threshold=(.85 * (1 - .85)))),
##    ('select_kbest', SelectKBest(chi2, k=500)),
    ('tfidf_transformer', TfidfTransformer()),
    ('classifier',  BernoulliNB()) ])
##    ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))  ])
##    ('classifier', SGDClassifier(loss='log'))  ])
##    ('classifier', RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0))  ])
##    ('classifier', DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0))  ])
##    ('classifier', KNeighborsClassifier(n_neighbors=3))  ])
##    ('classifier', BaggingClassifier(BernoulliNB(), max_samples=1.0, max_features=1.0))  ])
##    ('densifier', DenseTransformer()), ('classifier', GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=1, random_state=0))  ])

print("")

# k-fold cross validation
k = 6
kf = KFold(n_splits=k, random_state=None, shuffle=False)
scores = []
confusion = np.array([[0, 0], [0, 0]])
i = 1
sum_apy_ml = 0
sum_apy_index_fund = 0
for train_indices, test_indices in kf.split(data):
    print("Training and testing fold", str(i))
    
    train_text = data.iloc[train_indices]['features'].values
    train_y = data.iloc[train_indices]['class'].values

    test_text = data.iloc[test_indices]['features'].values
    test_y = data.iloc[test_indices]['class'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)    

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=ROSE)
    scores.append(score)

    # calculate profits from investing in each test set using predictions
    normalized_profit_index_fund = 0
    normalized_profit_ml = 0
    portfolio_size_ml = 0
    for index in test_indices:
        stock = data.iloc[index].name
        init_price = float(stored_data[stock][0])
        fin_price = float(stored_data[stock][1])
        normalized_profit_index_fund += (fin_price - init_price) / init_price
        prediction_index = np.where(test_indices == index)[0][0]
        if predictions[prediction_index] == 1:
            normalized_profit_ml += (fin_price - init_price) / init_price
            portfolio_size_ml += 1
        
    normalized_profit_index_fund = normalized_profit_index_fund / len(test_indices)
    annualized_profit_index_fund = (1 + normalized_profit_index_fund)**MARKET_PERIODS_PER_YEAR - 1
    normalized_profit_ml = normalized_profit_ml / portfolio_size_ml
    annualized_profit_ml = (1 + normalized_profit_ml)**MARKET_PERIODS_PER_YEAR - 1
    
    print('APY from machine learning strategy for set ' + str(i) + ':', annualized_profit_ml)
    print('APY from index fund (control strategy) for set ' + str(i) + ':', annualized_profit_index_fund)

    sum_apy_ml += annualized_profit_ml
    sum_apy_index_fund += annualized_profit_index_fund

    i += 1

avg_apy_ml = sum_apy_ml / k
avg_apy_index_fund = sum_apy_index_fund / k
print('')
print('Average APY from ML strategy:', avg_apy_ml)
print('Average APY from index fund strategy:', avg_apy_index_fund)
print('')

print('Total stocks classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
