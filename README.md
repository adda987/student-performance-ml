````markdown
#  OULAD ML Platform  
###  Platformă de Machine Learning pentru analiza performanței studenților

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Finalizat-brightgreen?style=for-the-badge"/>
</p>

---

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

##  Scopul proiectului

Acest proiect a fost realizat în cadrul lucrării de licență:

> **„Predicția performanței academice a studenților utilizând metode de Machine Learning și Learning Analytics”**

---

##  Tehnologii utilizate

* Python
* Streamlit
* Scikit-learn
* Pandas / NumPy
* Plotly / Altair / Matplotlib



**Andreea S.**
Informatică Economică — ASE București
