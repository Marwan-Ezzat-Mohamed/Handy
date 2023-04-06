import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# precision_recall_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, accuracy_score
# import Model from keras
from keras.models import Model
import json
from main import load_features, create_model, LABEL_MAP_PATH


def plot_confusion_matrix(model, X_test, y_test, label_map, indices=None):
    # Generate predictions on the test set
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    # Filter the predictions and true labels based on the specified indices
    if indices is not None:
        y_pred = y_pred[indices]
        y_true = y_true[indices]

    # Get the relevant actions (those that are present in y_true)
    relevant_actions = []
    for i in y_true:
        if label_map[str(i)] not in relevant_actions:
            relevant_actions.append(label_map[str(i)])

    # Calculate the confusion matrix for the relevant actions
    cm = confusion_matrix(y_true, y_pred, labels=list(
        range(len(relevant_actions))))

    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(relevant_actions)), relevant_actions, rotation=90)
    plt.yticks(np.arange(len(relevant_actions)), relevant_actions)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


def plot_action_accuracy(model, X_test, y_test, actions):
    # Generate predictions on the test set
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    # Calculate the accuracy for each action
    action_acc = {}
    for i, action in enumerate(actions):
        action_indices = np.where(y_true == i)[0]
        action_acc[action] = accuracy_score(
            y_true[action_indices], y_pred[action_indices])

    action_acc_map = sorted(
        action_acc.items(), key=lambda x: x[1], reverse=True)

    # display the top 50 actions with highest accuracy
    # print only the action name

    print(action_acc_map)

    # Plot the accuracy for each action
    plt.bar(actions, action_acc.values())
    plt.title('Model Accuracy by Action')
    plt.xlabel('Action')
    plt.ylabel('Accuracy')
    plt.show()


def main():
    # Load the data
    label_map = {}
    with open(LABEL_MAP_PATH) as fp:
        label_map = json.load(fp)
    actions = np.array(list(label_map.values()))

    print("actions: ", actions)

    # X_train, y_train = load_features(actions, label_map, data_type='train')
    X_test, y_test = load_features(actions, label_map, data_type='test')

    # view model arch using visualkeras
    from visualkeras import layered_view

    # Create the model
    model = create_model(actions)
    # layered_view(model, legend=True, to_file='model.png')

    plot_action_accuracy(model, X_test, y_test, actions)

    # y_pred = np.argmax(model.predict(X_test), axis=-1)
    # # get the accuracy of each action in the test set
    # accuracy_map = {}
    # for i, action in enumerate(actions):
    #     action_indices = np.where(y_test == i)[0]
    #     accuracy_map[action] = accuracy_score(

    # # print top 50 actions with highest accuracy
    # sorted_accuracy_map = sorted(
    #     accuracy_map.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_accuracy_map[:50])
    # # Train the model
    # history = model.fit(X_train, y_train, validation_data=(
    #     X_test, y_test), epochs=10, batch_size=32)

    # Plot the confusion matrix
    plot_confusion_matrix(model, X_test, y_test,
                          label_map, indices=range(200))


if __name__ == '__main__':
    main()
