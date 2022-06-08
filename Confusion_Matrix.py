from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd

actual_array = ["2", "1", "1", "1", "4", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "3", "4", "4", "1", "1", "1", "4", "4",
                "2", "2", "1", "1", "4", "4", "4", "4", "Null", "Null", "1", "1", "3", "1", "1", "1", "1", "1", "1", "1", "1", "4", "4", "1", "1", "1", "1", "1", "1", ]

predict_array = ["2", "Null", "2", "1", "4", "1", "2", "Null", "Null", "1", "2", "1", "Null", "Null", "2", "2", "2", "1", "Null", "1", "1", "Null",
                 "Null", "Null", "1", "2", "3", "3", "4", "Null", "2", "2", "2", "4", "1", "Null", "Null", "2", "Null", "2", "2", "2", "2", "2", "1", "2",
                 "2", "3", "1", "1", "1", "1", "1", "1", "2", "4", "4", "4", "1", "2", "2", "2", "1", "1"]

actual_array_sum = len(actual_array)
predict_array_sum = len(predict_array)

y_actual = pd.Series(actual_array, name='Actual')
y_predict = pd.Series(predict_array, name='Predicted')

df_classification_report = classification_report(y_predict, y_actual)
df_confusion = pd.crosstab(y_predict, y_actual)

print(f'Actual array sum :{actual_array_sum}\nPredict array sum: {predict_array_sum}\n')
print(f'{df_classification_report}\n')
print(f'{df_confusion}\n')

# print(df_confusion)
# print(df_classification_report)
# print(f'Accuracy: {accuracy_score(y_actual, y_predict)}\n')
