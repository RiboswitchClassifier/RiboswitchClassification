import fasttext

# 4 Way Classifier
# thread = 4, dim = 400, epoch = 15000 or 8000, lr = 0.1

# 16 Way Classifier
# thread = 4, dim = 300, epoch = 25, lr = 0.1 => 80%
# thread = 4, dim = 400, epoch = 1000, lr = 0.1 => 80%
# thread = 8, dim = 700, epoch = 50, lr = 0.5 => 80%


# 16 Riboswitch Classes : Classifier 
# Sample Size : 36,250
# trained on 30,000 ribo sequences
# Validated on 6,250

# Build New Model
# classifier = fasttext.supervised('FastTextDatasets/Ribo16WayDatasetS.train', 'FastTextDatasets/model', label_prefix='__label__', thread = 8, dim = 300, epoch = 25, lr = 0.5)

# Use already Built Model
classifier = fasttext.load_model('FastTextDatasets/model.bin', label_prefix ='__label__')

result = classifier.test('FastTextDatasets/Ribo16WayDatasetS.valid')
print ('P@1:', result.precision)
print ('R@1:', result.recall)
print ('Number of examples:', result.nexamples)
