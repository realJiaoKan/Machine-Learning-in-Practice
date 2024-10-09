
import os
import sys
from io import open

# for data and saves
import pandas as pd
import numpy as np
import dill
from PIL import Image # pillow package

# for app
import streamlit as st


# paths
path_to_repo = os.path.dirname(os.getcwd())
path_to_data = os.path.join(path_to_repo, 'data', 'tmp', 'bulldozers-raw')



# custom package
sys.path.insert(0, os.path.join(path_to_repo, 'src'))
from emlyon.utils import *






#**********************************************************
#*                      functions                         *
#**********************************************************
def display_bulldozer_img(index):
    st.subheader('Selected bulldozer')
    empty1, col, empty2 = st.columns([0.25, 0.5, 0.25])
    ind = index % len(st.session_state.imgs) # congruence
    st.write('The result of {} modulo {} is {}'.format(
        index, 
        len(st.session_state.imgs),
        ind,
    ))
    img = st.session_state.imgs[ind]
    col.image(img, use_column_width = 'always')
    return


def display_bulldozer_price(index):
    # compute model prediction
    pred_price = st.session_state.model.predict([st.session_state.X.values[index]])[0]
    pred_price = int(np.exp(pred_price))
    true_price = int(np.exp(st.session_state.y[index]))

    # display actual and predicted prices
    col_price, col_pred = st.columns(2)
    with col_price:
        st.subheader('real price')
        st.write(str(true_price) + ' Euros')
    with col_pred:
        st.subheader('estimated price')
        st.write(str(pred_price) + ' Euros')
    return


def display_bulldozer_features(index):
    st.subheader('Bulldozer features')
    feat0, val0, feat1, val1 = st.columns([3.5, 1.5, 3.5, 1.5])
    row = st.session_state.X.values[index]
    for i, feature in enumerate(st.session_state.X.columns):
        ind = i % 2
        if ind == 0:
            with feat0:
                st.warning(feature)
            with val0:
                st.info(str(row[i]))
        elif ind == 1:
            with feat1:
                st.warning(feature)
            with val1:
                st.info(str(row[i]))
    return



#**********************************************************
#                     main script                         *
#**********************************************************


#st.sidebar.title('Bulldozer _viewer_')
#st.title('Bulldozer _viewer_')

# session state
if 'model' not in st.session_state:
    # validation set given in notebook
    n_valid = 12000

    # import raw data
    df_raw = pd.read_feather(path_to_data)

    # preprocess data
    X, y, nas = proc_df(df_raw, 'SalePrice')
    X, y = X[n_valid:], y[n_valid:]

    # load regression model
    path_to_model = os.path.join(path_to_repo, 'saves', 'RF_regressor.pk')
    with open(path_to_model, 'rb') as file:
        model = dill.load(file)

    # load bulldozer images
    imgs = []
    path_to_imgs = os.path.join(path_to_repo, 'streamlit', 'img')
    img_files = os.listdir(path_to_imgs)
    for img_file in img_files:
        img = Image.open(os.path.join(path_to_imgs, img_file))
        img = np.array(img)
        imgs.append(img)

    # store in cache
    st.session_state.n_valid = n_valid
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.model = model
    st.session_state.imgs = imgs




st.title('Choose a bulldozer')
st.write(' ')
st.write(' ')

# exo 1:
# use st.selectbox in order to choose an index among bulldozer_list
bulldozer_list = ['-'] + list(range(1, st.session_state.n_valid + 1))

index = st.selectbox(
	label = 'Choose a bulldozer index',
	options = bulldozer_list,
	index = 0,
)
if index != '-':
    st.write('You selected option {}'.format(index))
	
	# exo 2 : display a bulldozer image
    display_bulldozer_img(index)
	
    display_bulldozer_price(index)

    display_bulldozer_features(index)
