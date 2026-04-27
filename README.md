#  OULAD ML Platform  
##  Platformă Machine Learning pentru analiza performanței studenților



##  Descriere

Acest proiect reprezintă o **platformă interactivă de Machine Learning** dezvoltată pentru analiza și predicția performanței studenților, utilizând:

>  **Open University Learning Analytics Dataset (OULAD)**

###  Aplicația integrează:
-  preprocesare completă a datelor  
-  modele de regresie  
-  modele de clasificare  
-  teste econometrice  
-  vizualizări interactive (Streamlit)  

---

##  Modele implementate

###  Regresie
- OLS (Regresie liniară)
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- SVR (Support Vector Regression)

###  Clasificare
- Random Forest Classifier
- SVM Classifier
- Logistic Regression
- Decision Tree
- KNN Classifier

---

##  Structura proiectului

```bash
oulad-ml-platform/
│
├── Home.py
├── utils.py
├── requirements.txt
├── oulad_students.csv
│
├── pages/
│   ├── 1_OLS.py
│   ├── 2_Ridge.py
│   ├── 3_Lasso.py
│   ├── 4_RF_Regressor.py
│   ├── 5_SVR.py
│   ├── 7_RF_Classifier.py
│   └── ...
````

---

##  Instalare și rulare

```bash
git clone https://github.com/USERNAME/oulad-ml-platform.git
cd oulad-ml-platform
pip install -r requirements.txt
streamlit run Home.py
```

---

##  Set de date

*  Sursă: **OULAD (Open University Learning Analytics Dataset)**
*  Număr studenți: ~32.000
*  Variabile: demografice, activitate, performanță academică

---


##  Tehnologii utilizate

* Python
* Streamlit
* Scikit-learn
* Pandas / NumPy
* Plotly / Altair / Matplotlib




**Andreea S.**

Informatică Economică — ASE București
