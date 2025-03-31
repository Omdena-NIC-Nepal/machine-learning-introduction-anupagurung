{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "dd21dee5-5f0a-498b-9024-208672d9a729",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import joblib\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a4254ce1-70c2-4145-89ab-15a110129762",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 23.29993210200506\n",
      "R-squared: 0.635285833864154\n"
     ]
    }
   ],
   "source": [
    "def train_model(data_path, model_path):\n",
    "    data = np.load(data_path)\n",
    "    X_train, X_test, y_train, y_test = data[\"X_train\"], data[\"X_test\"], data[\"y_train\"], data[\"y_test\"]\n",
    "    \n",
    "    model = LinearRegression()\n",
    "    model.fit(X_train, y_train)\n",
    "    \n",
    "    joblib.dump(model, model_path)\n",
    "    \n",
    "    y_pred = model.predict(X_test)\n",
    "    \n",
    "    mse = mean_squared_error(y_test, y_pred)\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    \n",
    "    print(f\"Mean Squared Error: {mse}\")\n",
    "    print(f\"R-squared: {r2}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    train_model(\"../data/preprocessed_data.npz\", \"../models/linear_regression.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bb3ef3aa-37fe-4e0d-a655-79d6dd58e1f5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
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
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
