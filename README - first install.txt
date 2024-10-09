1) get anaconda installed
2) create a new conda environment called 'mlp-course',
in a conda prompt run :
conda create -n mlp-course pip python=3.8

(to remove the env: conda remove -n mlp-course --all)

3) activate the env : conda activate mlp-course

4) install all dependencies:
cd path/to/repo
pip install -r requirements.txt

5) launch your jupyter notebook within this new environment, 
execute: jupyter notebook

For this Machine Learning Course the dataset we consider is the bluebook for bulldozers 
competition: https://www.kaggle.com/c/bluebook-for-bulldozers/overview
You can download the Train.zip data file here: 
https://www.kaggle.com/c/bluebook-for-bulldozers/data, 
and move unzip your file in the subfolder 
data/bulldozers


run the streamlit app:
cd path/to/app.py
streamlit run app.py