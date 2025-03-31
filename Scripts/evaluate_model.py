{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7a2ee187-4b0f-48a5-b1b3-6f77ddf13eca",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import joblib\n",
    "from sklearn.metrics import mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "54bff0a5-64d1-4eb3-b9a2-bbc7e03217c1",
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
    "def evaluate_model(data_path, model_path):\n",
    "    data = np.load(data_path)\n",
    "    X_test, y_test = data[\"X_test\"], data[\"y_test\"]\n",
    "\n",
    "    model = joblib.load(model_path)\n",
    "    y_pred = model.predict(X_test)\n",
    "\n",
    "    mse = mean_squared_error(y_test, y_pred)\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "    print(f\"Mean Squared Error: {mse}\")\n",
    "    print(f\"R-squared: {r2}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    evaluate_model(\"../data/preprocessed_data.npz\", \"../models/linear_regression.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bea96cfc-a131-4f7c-b298-c1587f7587d4",
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
