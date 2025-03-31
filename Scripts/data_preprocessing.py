{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "03bd6f30-a8ff-498c-8c69-013ed0146bef",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from scipy import stats\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d44ca50d-2180-4970-9703-662cdbd7a18a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_data(input_file, output_file):\n",
    "    df = pd.read_csv(input_file)\n",
    "    df.fillna(df.mean(), inplace=True)\n",
    "    df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]\n",
    "    \n",
    "    X = df.drop(columns=['MEDV'])\n",
    "    y = df['MEDV']\n",
    "    \n",
    "    scaler = StandardScaler()\n",
    "    X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
    "\n",
    "    preprocessed_data = {\"X_train\": X_train, \"X_test\": X_test, \"y_train\": y_train, \"y_test\": y_test}\n",
    "    np.savez(output_file, **preprocessed_data)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    preprocess_data(r\"C:\\Users\\User\\Documents\\machine-learning-introduction-anupagurung\\Data\\HousingData.csv\", \"../data/preprocessed_data.npz\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29594cb6-c801-4f3d-9bbf-7fbfebccd6a7",
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
