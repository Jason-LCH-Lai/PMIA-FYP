from sklearn.linear_model import LogisticRegression
from mia import *
from datasets.synthetic_dataset import *

# TODO: Test with other classifiers. Currently we only use RF for everything.


# Make sure RF is registered as a classifier with posteriors.
class LR(LogisticRegression, ClassifierWithPosteriors):
    def __init__(self, random_state=0, multi_class='multinomial', solver='lbfgs'):
        super(LR, self).__init__(random_state=random_state, multi_class=multi_class, solver=solver, max_iter=1000)


features = np.load('partial-membership-main/datasets/modified_purchase100.npy', allow_pickle=True)
labels = np.load('partial-membership-main/datasets/cluster_labels_10.npy', allow_pickle=True)

features_list = features.tolist()
labels_list = labels.tolist()


# Create a random dataset.
dataset = LabeledDataset(
  features_list['features'],labels
)

# Initialise a sensible MIA config.
config = MIAConfig.from_proportions(dataset.len(), 0.5, 1000, 499)

# Initialise the MIA environment.
env = MIAEnvironment(
    dataset,
    LR(random_state=0),
    LR(random_state=0),
    10,
    LR(random_state=0),
    config,
)

print("Training basic models")
env.train_basic_models()

print("Testing basic models")
print(env.test_basic_models())

print("Training attack models")
env.train_attack_models()

print("Testing attack models")
print(env.test_attack_models())

print("Test evaluation method")
test_size = 100
print("Results for training data")
training_samples = env.target_data.training_data.X[:test_size]
print(env.is_member(training_samples))

print("Results for non-training data")
testing_samples = env.target_data.testing_data.X[:test_size]
print(env.is_member(testing_samples))
