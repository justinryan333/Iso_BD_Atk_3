# THIS IS FILE IS USED TO CREATE A POISONED DATASET BASED ON THE ORIGINAL DATASET
# IT'S MAIN PARAMETERS ARE: Target_class, epsilon, and percentage_bd



# Importing the necessary libraries
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_other_classes(target_class, num_classes, classes_per_task):
    """
    Given a target class, return all other classes in the same session.

    Parameters:
    - target_class (int): The selected target class.
    - num_classes (int): Total number of classes.
    - classes_per_task (int): Number of classes per session/task.

    Returns:
    - List[int]: A list of other class indices in the same session.
    """
    # Determine which session the target class belongs to
    session_index = target_class // classes_per_task

    # Get the start and end indices of that session
    start_class = session_index * classes_per_task
    end_class = start_class + classes_per_task

    # Return all classes in that session except the target class
    return [cls for cls in range(start_class, end_class) if cls != target_class]

def get_subset_cifar10(dataset, num_bd, classes_taken, seed=None):
    """
    Create a subset of the CIFAR-10 dataset by selecting a fixed number of images (num_bd)
    from each class in classes_taken.

    Parameters:
    - dataset (Dataset): The CIFAR-10 dataset.
    - num_bd (int): Number of images to take from each selected class.
    - classes_taken (list): List of class labels to include in the subset.
    - seed (int, optional): Seed for random number generator.

    Returns:
    - Subset: A subset of the CIFAR-10 dataset containing num_bd images from each selected class.
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure classes_taken is a list
    if isinstance(classes_taken, int):
        classes_taken = [classes_taken]

    # Initialize list to store selected indices
    selected_indices = []

    # Iterate over the selected classes
    for class_label in classes_taken:
        # Get indices of images belonging to the current class
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_label]

        # Ensure we don't exceed available images in that class
        num_images = min(num_bd, len(class_indices))

        # Randomly select num_images from the class
        selected_indices.extend(np.random.choice(class_indices, int(num_images), replace=False))

    # Create and return the subset
    return Subset(dataset, selected_indices)

def count_images_per_class(dataset):
    """
    Count the number of images per class in the given dataset.

    Parameters:
    dataset (Dataset): The dataset to count images in.

    Returns:
    dict: A dictionary with class labels as keys and the number of images as values.
    """
    class_counts = {i: 0 for i in range(10)}

    for _, label in dataset:
        class_counts[int(label)] += 1

    return class_counts


def poison_images_with_CV2(dataset, target_class, epsilon):
    """
    Poison a set of images by adding a rectangle and assign them a new label of the target class using OpenCV.

    Parameters:
    dataset (Dataset): The dataset to poison.
    target_class (int): The new label to assign to poisoned images.
    epsilon (float): The poisoning parameter to apply to images.

    Returns:
    Dataset: A new dataset with poisoned images and updated labels.
    """
    poisoned_data = []
    poisoned_labels = []

    for image, _ in dataset:
        # Convert the image to a numpy array
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))
        # Draw a rectangle on the image
        image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)
        # Apply poisoning transformation
        image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)
        # Convert the poisoned image back to a tensor
        poisoned_image = torch.tensor(np.transpose(image_np_HWC_poison, (2, 0, 1)))

        poisoned_data.append(poisoned_image)
        poisoned_labels.append(target_class)
    poisoned_dataset = torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))
    return poisoned_dataset


def poison_images_with_CV2_v2(dataset, target_class, epsilon):
    """
    Poison a set of images from a specific class while preserving their original labels.
    The original dataset remains unchanged. The poisoned dataset has only the poisoned
    images of the attacked class, while others remain unchanged.

    Parameters:
    dataset (Dataset): The dataset to poison.
    target_class (int): The class to poison.
    epsilon (float): The poisoning parameter to apply to images.

    Returns:
    (original_dataset, poisoned_dataset): A tuple containing the original dataset and the poisoned dataset.
    """
    original_data = []
    original_labels = []
    poisoned_data = []
    poisoned_labels = []

    for image, label in dataset:
        # Append the original image and label to the original dataset
        original_data.append(image)
        original_labels.append(label)

        # Convert the image to a numpy array (HWC format)
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))

        # Apply poisoning only to images of the target class
        if label == target_class:
            # Draw a rectangle on the image (Poisoning step)
            image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)
            # Apply poisoning transformation
            image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)
            # Convert the poisoned image back to a tensor
            poisoned_image = torch.tensor(np.transpose(image_np_HWC_poison, (2, 0, 1)))
        else:
            poisoned_image = image  # Keep original image if not the target class

        # Append the poisoned image and its original label to the poisoned dataset
        poisoned_data.append(poisoned_image)
        poisoned_labels.append(label)

    # Create the original dataset (unchanged)
    original_dataset = torch.utils.data.TensorDataset(torch.stack(original_data), torch.tensor(original_labels))
    # Create the poisoned dataset (with poisoned images for target class)
    poisoned_dataset = torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))

    return original_dataset, poisoned_dataset









# MAIN CODE BLOCK

# -----------------------------------------
# Step 0: preperations
# initialize/set parameters
# set the seed for reproducibility
# create transformation
# Load the CIFAR-10 datasets
# Set Run Mode
# -----------------------------------------
num_classes = 10  # The total number of classes in the dataset.
classes_per_task = 2  # The number of classes in each task
target_class = 4
other_classes = get_other_classes(target_class, num_classes, classes_per_task)
print(f"Target Class: {target_class}, Other Classes in Session: {other_classes}")

# NOTE: classes and task. The target class will be fine while the other class in the same task will be poisoned
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#              0      1       2      3       4      5      6       7         8      9
#             Task 1: 0-1, | task 2:2-3, | task 3: 4-5,  | task 4: 6-7,  |  task 5: 8-9

epsilon = 0.05  # The epsilon value for the poisoning attack
percentage_bd = 0.05  # The percentage of images from the dataset to be poisoned
num_bd = 5000*percentage_bd  # The number of images to be poisoned
print(f'Number of images to be poisoned: {num_bd}')
print('___________________________________________________________________________________________')
print()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU

# Load the CIFAR-10 training datasets
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


is_testing = True # Set to True to run the testing code, False to run the training code. This turns on or off print statements and some calcualtions










# -----------------------------------------
# Step 1: Create the training dataset
# 1.1: Prepare the CIFAR-10 dataset (The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class.)
# 1.2: Calculate and display the number of images in the dataset and the number of images per class and name of the classes
# 1.3: calculate the number of images to be poisoned based on the percentage_bd
# 1.4: create a subset of the dataset of images taken. The subset will be used to create the poisoned dataset
# 1.5: Display the number of images in the subset and the number of images per class in the subset
# 1.6 poison the images in the subset
# 1.6.1: apply poison pattern to the images in the subset
# 1.6.2: change the label of the images in the subset to the target class
# 1.7: Display the number of images in the poisoned subset and the number of images per class in the poisoned subset
# 1.8: append the poisoned subset to the original dataset
# 1.9: Display the number of images in the new dataset and the number of images per class in the new dataset
# -----------------------------------------
print('Part 1: Training Set Creation')
# count the number of images in the original train dataset
print(f'Number of images in original train dataset: {len(train_set):,}')
# Count the number of images per class in the new sub-dataset
class_counts = count_images_per_class(train_set)  # Assuming this returns a dictionary
# Print the number of images per class
print(f'Number of images per class in original train dataset:', end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()

# Create a subset of the training dataset
subset_train_set = get_subset_cifar10(train_set, num_bd, other_classes, seed=seed)
print(f'Number of images in the subset train dataset: {len(subset_train_set):,}')
# Count the number of images per class in the new sub-dataset
class_counts = count_images_per_class(subset_train_set)  # Assuming this returns a dictionary
print(f'Number of images per class in the subset train dataset:', end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()


# Check that orignal dataset is not changed
print('Number of images in the original train dataset after creating the subset:', len(train_set))

# Poison the subset
poisoned_subset_train_set = poison_images_with_CV2(subset_train_set, target_class, epsilon)

# Count the number of images per class in the poisoned subset
class_counts = count_images_per_class(poisoned_subset_train_set)  # Assuming this returns a dictionary
print(f'Number of images in the poisoned subset train dataset: {len(poisoned_subset_train_set):,}')
print(f'Number of images per class in the poisoned subset train dataset:', end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()


# display a sample of the poisoned images
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    image, _ = poisoned_subset_train_set[i]
    axs[i].imshow(image.permute(1, 2, 0))
    axs[i].axis('off')
plt.show()

# Append the poisoned subset to the original training dataset
new_train_set = torch.utils.data.ConcatDataset([train_set, poisoned_subset_train_set])

# Count the number of images per class in the new training dataset
class_counts = count_images_per_class(new_train_set)  # Assuming this returns a dictionary
print(f'Number of images in the new train dataset: {len(new_train_set):,}')
print(f'Number of images per class in the new train dataset:', end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()


print('___________________________________________________________________________________________')











# -----------------------------------------
#Step 2: create the test dataset
# 2.1: Load the CIFAR-10 test dataset
# 2.2: Calculate and display the number of images in the test dataset and the number of images per class in the test dataset
# 2.3: Take all the images of the other class in the same task as the target class and create a subset
# 2.4: poison the images in the subset
# 2.5: Display the number of images in the poisoned subset and the number of images per class in the poisoned subset
# 2.6: append the poisoned subset to the original test dataset
# 2.7: Display the number of images in the new test dataset and the number of images per class in the new test dataset
# 1.8: USE THIS POISONED SUBSET ONLY DURING  the testing of the after all training is done
print('Part 2: test Set Creation')

# Print the number of images in the original test dataset
print(f'Number of images in the original test dataset: {len(test_set):,}')

# Count the number of images per class in the new test dataset
class_counts = count_images_per_class(test_set)  # Assuming this returns a dictionary

# Print the number of images per class
print(f'Number of images per class in the original test dataset:',end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()

#create new test dataset using poison_images_with_CV2_v2
test_set, test_set_poisoned = poison_images_with_CV2_v2(test_set, other_classes, epsilon=0.5)

# Count the number of images per class in the poisoned test dataset
class_counts = count_images_per_class(test_set_poisoned)  # Assuming this returns a dictionary
print(f'Number of images in the poisoned test dataset: {len(test_set_poisoned):,}')
print(f'Number of images per class in the poisoned test dataset:', end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()

# Classes to display
from matplotlib.widgets import Button

# Classes to display
classes_to_show = [2, 3, 4, 5]

# Initial index
current_index = 0

# Create subplots with 2 rows (one for original, one for poisoned) and len(classes_to_show) columns
fig, axs = plt.subplots(2, len(classes_to_show), figsize=(20, 6))


# Function to update the images
def update_images(index):
    for i, class_id in enumerate(classes_to_show):
        # Find an image from the poisoned dataset for the current class
        count = 0
        for image, label in test_set_poisoned:
            if label == class_id:
                if count == index:
                    axs[0, i].imshow(image.permute(1, 2, 0), interpolation='nearest')  # Show image as large as possible
                    axs[0, i].axis('off')  # Hide axes
                    break
                count += 1

        # Find an image from the original dataset for the current class
        count = 0
        for image, label in test_set:
            if label == class_id:
                if count == index:
                    axs[1, i].imshow(image.permute(1, 2, 0), interpolation='nearest')  # Show image as large as possible
                    axs[1, i].axis('off')  # Hide axes
                    break
                count += 1

    # Update titles
    axs[0, 0].set_title(f"Poisoned Dataset (Image {index + 1})", fontsize=16, pad=20)  # Title for the first row
    axs[1, 0].set_title(f"Original Dataset (Image {index + 1})", fontsize=16, pad=20)  # Title for the second row
    plt.draw()


# Next image button function
def next_image(event):
    global current_index
    current_index += 1
    update_images(current_index)


# Previous image button function
def prev_image(event):
    global current_index
    if current_index > 0:
        current_index -= 1
        update_images(current_index)


# Create buttons for navigation
ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])  # Position for next button
button_next = Button(ax_next, 'Next Image')
button_next.on_clicked(next_image)

ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])  # Position for previous button
button_prev = Button(ax_prev, 'Previous Image')
button_prev.on_clicked(prev_image)

# Initialize the first image
update_images(current_index)

# Adjust layout for better spacing
plt.subplots_adjust(wspace=0.1, hspace=0.5)

# Display the plot
plt.show()



print('___________________________________________________________________________________________')
# -----------------------------------------


# -----------------------------------------



print('CODE DONE')
