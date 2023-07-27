Today, I was handling some #timeseriesanalysis tasks and I realized that the different forecasting strategies in the #sktime python library yielded widely varying values. 
Different #sktime forecasting strategies have different strengths and weaknesses and may perform better or worse depending on the characteristics of the data. 
Thus, I did a cross-validation of strategies as the best approach to determine the most appropriate #forecasting strategy and selected the one that provided the most 'accurate' forecast; 
as in the one with the least mean absolute error (MAE) value.
