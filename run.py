from helpers import load_training, create_submission
from fcn_model import FCNModel

# Choose if the model has to be fully trained or can be restored
restore = True
# Training path, without trailing slash
training_path = 'training'
# Testing path, without trailing slash
test_path = 'test_set_images'

# Load the image
imgs, gt_imgs = load_training(training_path)

# Create the model
fcnModel = FCNModel()

# Fit the model with the training data
fcnModel.fit(imgs, gt_imgs, augment=True, dropout=True, weights='imagenet', restore=restore,
            num_batches=2, steps_per_epoch=64, epochs=5, validation_steps=32)

# Create the submission from the model
create_submission(test_path, 'submission.csv', fcnModel)
