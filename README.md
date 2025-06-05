# jsm-learn
### A Python library for data analysis using JSM-method 
## Introduction
### The JSM-method
**The JSM-method for automatised hypothesis forming** (rus. *ДСМ-метод автоматического порождения гипотез, ДСМ-АПГ*) is a data analysis method first introduced by a group of researchers lead by Viktor Finn (Виктор Константинович Финн) in 1983. The JSM-method is not a probability-based data analysis method; instead it is based on mathematical logic. This allows the JSM-method to not only predict values based on reasoning-like procedures, but also to provide explanations on why those predictions where made. It has been successfully used in many studies, proving itself particularly useful in fields like sociology, medicine, pharmacology and criminology.
### The `jsm-learn` package
The `jsm-learn` package provides a version of the JSM-method using set theory. This package contains a `JSM` class which can be used to instantiate a model applying the JSM-method. The package was designed to be easy to approach for users familiar with the `scikit-learn` package.
## Quick-start guide
1. Import the module:
    ```python
    import jsm-learn
    ```
2. Instantiate a new model:
    ```python
    my_jsm = JSM()
    ```
3. Fit the model to your data:
    ```python
    my_jsm.fit(training_data, target_data)
    ```
4. Run the model:
    ```python
    my_jsm.predict()
    ```
5. View the results as a `pandas` dataframe:
    ```python
    results = my_jsm.to_df()
    ```

## Documentation

## Acknowledgments
This package was developed by Mikhail Torkanovskiy for his bachelor's degree. The author would like to thank the following people:
* **Oleg Mikhaylovich Anshakov** for his help and support as the research supervisor;
* **Viktor Konstantinovich Finn** for his belief in the author and for inspiring him and many others with his neverending scientific vigor. 