from flask import Flask, jsonify, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

fruitData = pd.read_csv('fruits.csv')

X = fruitData.drop(columns=["fruit"])
Y = fruitData['fruit']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
p = model.predict(X_test)
accuracy = "{} %".format(accuracy_score(Y_test, p) * 100)


def getBinaryGender(gender):
    if (gender.lower() == "male"):
        return 1
    elif (gender.lower() == "female"):
        return 0
    else:
        raise Exception("Please enter either male or female as gender")


@app.route('/guess-fave-fruit', methods=['POST'])
def guess_fave_fruit():
    requestBody = request.get_json()

    age = requestBody['age']
    gender = getBinaryGender(requestBody["gender"])

    prediction = model.predict([[age, gender]])
    return jsonify({"prediction": prediction[0], "accuracy": accuracy})


if __name__ == "__main__":
    app.run()
