from sklearn.linear_model import LogisticRegression
from mia import *
from datasets.synthetic_dataset import *
from partial_mia import *
import random

# TODO: Test with other classifiers. Currently we only use RF for everything.


# Make sure RF is registered as a classifier with posteriors.
class LR(LogisticRegression, ClassifierWithPosteriors):
    pass

features = np.load('partial-membership-main/datasets/modified_purchase100.npy', allow_pickle=True)
labels = np.load('partial-membership-main/datasets/cluster_labels_20.npy', allow_pickle=True)

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
test_size = 150
print("Results for training data")
training_samples = env.target_data.training_data.X[:test_size]
print(env.is_member(training_samples))

print("Results for non-training data")
testing_samples = env.target_data.testing_data.X[:test_size]
print(env.is_member(testing_samples))

print()
print("\nPartial MIA_1")
# Parameters remain the same as before
range_size = 10
r = np.linspace(0, 1, range_size)
column_number_1 = 0
column_number_2 = 1

print("Results for training data")
training_samples = env.target_data.training_data.X[:test_size]
training_product_dataset, (original_column1, original_column2) = product_dataset_modified(
    training_samples, r, column_number_1, r, column_number_2, return_original_columns=True
)

# Assuming 'env.is_member' can handle the modified dataset
is_member, membership_proba = env.is_member(training_product_dataset)
print("Membership decisions:")
print(is_member)
print("Membership probabilities:")
print(membership_proba)

# Correcting the print statement to handle two columns
print("Distance from original value for column 1:")
print(np.abs(training_product_dataset[:, column_number_1] - original_column1))
print("Distance from original value for column 2:")
print(np.abs(training_product_dataset[:, column_number_2] - original_column2))


print("Results for non-training data")
testing_samples = env.target_data.testing_data.X[:test_size]
testing_product_dataset, original_column = product_dataset_modified(
     testing_samples, r, column_number_1, r, column_number_2, return_original_columns=True
)
is_member, membership_proba = env.is_member(testing_product_dataset)
print("Membership decisions:")
print(is_member)
print("Membership probabilities:")
print(membership_proba)
print("Distance from original value for column 1:")
print(np.abs(testing_product_dataset[:, column_number_1] - original_column1))
print("Distance from original value for column 2:")
print(np.abs(testing_product_dataset[:, column_number_2] - original_column2))

print()
print("partial_MIA-gradually decrease")

print("Results for training data")
training_samples = env.target_data.training_data.X[:test_size]
train_features_to_select = int(0.7 * training_samples.shape[1])
train_first_data = training_samples[:, :train_features_to_select]
train_average_features = np.mean(train_first_data, axis=1, keepdims=True)
train_average_features_expanded = np.tile(train_average_features, (1, training_samples.shape[1]- train_features_to_select ))
train_validation_data = np.concatenate([train_first_data, train_average_features_expanded], axis=1)
train_is_member1, train_membership_proba1 = env.is_member(train_validation_data)
print("Membership decisions:")
print(train_is_member1)
print("Membership probabilities:")
print(train_membership_proba1)

print("Results for testing data")
testing_samples = env.target_data.testing_data.X[:test_size]
test_features_to_select = int(0.9 * testing_samples.shape[1])
test_first_data = testing_samples[:, :test_features_to_select]
test_average_features = np.mean(test_first_data, axis=1, keepdims=True)
test_average_features_expanded = np.tile(test_average_features, (1, testing_samples.shape[1]- test_features_to_select ))
test_validation_data = np.concatenate([test_first_data, test_average_features_expanded], axis=1)
test_is_member1, test_membership_proba1 = env.is_member(test_validation_data)
print("Membership decisions:")
print(test_is_member1)
print("Membership probabilities:")
print(test_membership_proba1)

print()
print("Partial MIA_3-Random Shuffle")

total_columns_train = training_samples.shape[1] 
half_columns_train = total_columns_train // 5 
column_numbers_to_permute_train = np.random.choice(range(total_columns_train), half_columns_train, replace=False).tolist()

print("Results for training data")
training_samples = env.target_data.training_data.X[:test_size]
permuted_training_samples = permute_features(training_samples, column_numbers_to_permute_train)
is_member, membership_proba = env.is_member(permuted_training_samples)
print("Membership decisions:")
print(is_member)
print("Membership probabilities:")
print(membership_proba)

total_columns_test = testing_samples.shape[1] 
half_columns_test = total_columns_test // 5 
column_numbers_to_permute_test = np.random.choice(range(total_columns_test), half_columns_test, replace=False).tolist()

print("Results for testing data")
testing_samples = env.target_data.testing_data.X[:test_size]
permuted_testing_samples = permute_features(testing_samples, column_numbers_to_permute_test)
is_member, membership_proba = env.is_member(permuted_testing_samples)
print("Membership decisions:")
print(is_member)
print("Membership probabilities:")
print(membership_proba)

print()
print("Partial MIA-Noisy")

print("Results for training data")
training_samples = env.target_data.training_data.X[:test_size]
total_columns_train1 = training_samples.shape[1] 
half_columns_train1 = total_columns_train1  
column_numbers_to_add_noise_train = np.random.choice(range(total_columns_train1), half_columns_train1, replace=False).tolist()

# Apply noise addition
noisy_training_samples = add_noise_to_features(training_samples, column_numbers_to_add_noise_train, noise_level=2)

# Evaluate the MIA on the dataset with added noise
is_member, membership_proba = env.is_member(noisy_training_samples)
print("Membership decisions:")
print(is_member)
print("Membership probabilities:")
print(membership_proba)

print("Results for testing data")
testing_samples = env.target_data.testing_data.X[:test_size]
total_columns_test1 = testing_samples.shape[1] 
half_columns_test1 = total_columns_test1 
column_numbers_to_add_noise_test = np.random.choice(range(total_columns_test1), half_columns_test1, replace=False).tolist()

noisy_testing_samples = add_noise_to_features(testing_samples, column_numbers_to_add_noise_test, noise_level=0.1)

is_member, membership_proba = env.is_member(noisy_testing_samples)
print("Membership decisions:")
print(is_member)
print("Membership probabilities:")
print(membership_proba)


