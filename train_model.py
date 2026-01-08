{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "26ab097d-dc02-4c6f-a8d7-a0a79228e942",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved successfully\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Bhavana\\AppData\\Local\\Temp\\ipykernel_15012\\1671873703.py:10: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  df['Age'].fillna(df['Age'].median(), inplace=True)\n",
      "C:\\Users\\Bhavana\\AppData\\Local\\Temp\\ipykernel_15012\\1671873703.py:11: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import joblib\n",
    "\n",
    "# Load data\n",
    "df = pd.read_csv(\"Titanic_train.csv\")\n",
    "\n",
    "# Data preprocessing\n",
    "df['Age'].fillna(df['Age'].median(), inplace=True)\n",
    "df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n",
    "df['Sex'] = df['Sex'].map({'male':0, 'female':1})\n",
    "df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)\n",
    "\n",
    "X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']]\n",
    "y = df['Survived']\n",
    "\n",
    "# Scaling\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Train model\n",
    "model = LogisticRegression()\n",
    "model.fit(X_scaled, y)\n",
    "\n",
    "# Save model and scaler\n",
    "joblib.dump(model, 'titanic_model.pkl')\n",
    "joblib.dump(scaler, 'scaler.pkl')\n",
    "\n",
    "print(\"Model saved successfully\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "135f9075-0fef-4dca-98d4-415ecb4aa0c3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:anaconda3] *",
   "language": "python",
   "name": "conda-env-anaconda3-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
